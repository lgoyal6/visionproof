"""Run the same evaluator on one short-lived AWS worker and terminate it.

Requires an authenticated AWS profile and an explicit output directory. COOL requires an
already entitled Marketplace AMI; this tool never buys/subscribes to a product. No inference
keys, AWS credentials, or unrelated files are transferred to the worker.
"""

import argparse
import json
import shlex
import subprocess
import tarfile
import time
import urllib.request
import uuid
from pathlib import Path

import boto3


def execute(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, capture_output=True, **kwargs)


def ssh_ready(connection):
    try:
        return subprocess.run(connection + ["test -f /var/tmp/opencv-proof-bootstrap-ready"],
                              capture_output=True, timeout=10).returncode == 0
    except subprocess.TimeoutExpired:
        return False


def bundle(project: Path, manifest: Path, output: Path):
    data = json.loads(manifest.read_text())
    with tarfile.open(output, "w:gz") as archive:
        archive.add(project / "pyproject.toml", arcname="project/pyproject.toml")
        for path in (project / "src").rglob("*.py"):
            archive.add(path, arcname="project/" + str(path.relative_to(project)))
        archive.add(manifest, arcname="data/manifest.json")
        for row in data["records"]:
            for field in ("image", "template", "label"):
                archive.add(manifest.parent / row[field], arcname="data/" + row[field])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True, help="frozen selection.json from local proof")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--target", choices=["arm-pip", "x86-pip", "cool"], required=True)
    parser.add_argument("--ami", help="required for COOL; must already be entitled")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--instance-type")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if args.target == "cool" and not args.ami:
        parser.error("COOL requires --ami; subscription is never automatic")
    kind = args.instance_type or ("c7i.large" if args.target == "x86-pip" else "c8g.large")
    config = json.loads(args.config.read_text())["config"]
    plan = {"region": args.region, "target": args.target, "instance_type": kind,
            "ami": args.ami, "maximum_worker_minutes": 40, "shutdown_behavior": "terminate",
            "config": config, "network": "SSH only from this client's public IPv4",
            "cost": "on-demand compute plus EBS/network and any existing COOL fee; no saving claim"}
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return
    session = boto3.Session(region_name=args.region)
    # Authentication is checked before creating any resource.
    session.client("sts").get_caller_identity()
    ec2 = session.client("ec2")
    architecture = "x86_64" if args.target == "x86-pip" else "arm64"
    if args.ami:
        image = ec2.describe_images(ImageIds=[args.ami])["Images"][0]
        if image["Architecture"] != architecture:
            raise ValueError("AMI architecture does not match target")
    else:
        images = ec2.describe_images(Owners=["099720109477"], Filters=[
            {"Name": "architecture", "Values": [architecture]},
            {"Name": "name", "Values": [f"ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-{'amd64' if architecture == 'x86_64' else 'arm64'}-server-*"]},
            {"Name": "state", "Values": ["available"]}])["Images"]
        image = max(images, key=lambda item: item["CreationDate"])
    subnets = ec2.describe_subnets(Filters=[{"Name": "default-for-az", "Values": ["true"]}])["Subnets"]
    offerings = ec2.describe_instance_type_offerings(LocationType="availability-zone", Filters=[
        {"Name": "instance-type", "Values": [kind]}])["InstanceTypeOfferings"]
    offered_zones = {item["Location"] for item in offerings}
    subnets = [subnet for subnet in subnets if subnet["AvailabilityZone"] in offered_zones]
    if not subnets:
        raise ValueError("no default subnet; supply infrastructure explicitly rather than creating a VPC")
    subnet = subnets[0]
    with urllib.request.urlopen("https://checkip.amazonaws.com", timeout=10) as response:
        import ipaddress
        client_ip = str(ipaddress.IPv4Address(response.read().decode().strip()))
    if args.out.exists() and any(args.out.iterdir()):
        raise ValueError("use a fresh output directory")
    args.out.mkdir(parents=True, exist_ok=True)
    project = Path(__file__).resolve().parents[1]
    archive = args.out / "worker-input.tar.gz"
    bundle(project, args.manifest, archive)
    (args.out / "config.json").write_text(json.dumps(config))
    token = "opencv-proof-" + uuid.uuid4().hex[:12]
    key = args.out / "worker-key"
    execute(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)])
    group_id, instance_id, imported = None, None, False
    resource = {**plan, "ami": image["ImageId"], "run_token": token}
    try:
        ec2.import_key_pair(KeyName=token, PublicKeyMaterial=key.with_suffix(".pub").read_bytes())
        imported = True
        group_id = ec2.create_security_group(GroupName=token, Description="Temporary OpenCV benchmark SSH",
                                             VpcId=subnet["VpcId"])["GroupId"]
        ec2.authorize_security_group_ingress(GroupId=group_id, IpPermissions=[{
            "IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
            "IpRanges": [{"CidrIp": client_ip + "/32"}]}])
        bootstrap = "#!/bin/bash\nset -eu\nshutdown -h +40\napt-get update -qq\nDEBIAN_FRONTEND=noninteractive apt-get install -y -qq python3-venv python3-pip\ntouch /var/tmp/opencv-proof-bootstrap-ready\n"
        instance = ec2.run_instances(ImageId=image["ImageId"], InstanceType=kind, MinCount=1, MaxCount=1,
            KeyName=token, ClientToken=token, UserData=bootstrap,
            InstanceInitiatedShutdownBehavior="terminate",
            MetadataOptions={"HttpTokens": "required", "HttpEndpoint": "enabled"},
            NetworkInterfaces=[{"DeviceIndex": 0, "SubnetId": subnet["SubnetId"],
                                "Groups": [group_id], "AssociatePublicIpAddress": True}],
            TagSpecifications=[{"ResourceType": "instance", "Tags": [
                {"Key": "Name", "Value": token}, {"Key": "Purpose", "Value": "opencv-deploy-proof"}]}])["Instances"][0]
        instance_id = instance["InstanceId"]
        resource.update(instance_id=instance_id, security_group_id=group_id)
        (args.out / "resources.json").write_text(json.dumps(resource, indent=2))
        print(json.dumps({"launched": instance_id, "target": args.target}), flush=True)
        deadline = time.monotonic() + 600
        address = None
        while time.monotonic() < deadline:
            try:
                current = ec2.describe_instances(InstanceIds=[instance_id])["Reservations"][0]["Instances"][0]
            except ec2.exceptions.ClientError as error:
                # EC2 RunInstances can return before DescribeInstances sees the new ID.
                if error.response["Error"]["Code"] != "InvalidInstanceID.NotFound":
                    raise
                time.sleep(2)
                continue
            address = current.get("PublicIpAddress")
            if address:
                connection = ["ssh", "-i", str(key), "-o", "StrictHostKeyChecking=accept-new",
                              "-o", "UserKnownHostsFile=" + str(args.out / "known_hosts"),
                              "-o", "ConnectTimeout=5", "ubuntu@" + address]
                if ssh_ready(connection):
                    break
            time.sleep(5)
        else:
            raise TimeoutError("worker bootstrap did not finish")
        scp = ["scp", "-i", str(key), "-o", "UserKnownHostsFile=" + str(args.out / "known_hosts")]
        execute(scp + [str(archive), str(args.out / "config.json"), f"ubuntu@{address}:/home/ubuntu/"])
        cool = args.target == "cool"
        install = ("test -x /opt/cool/venvs/python_3.12/bin/python; "
                   "export LD_LIBRARY_PATH=/opt/cool/cpp_sdk/lib:${LD_LIBRARY_PATH:-}; "
                   "export PYTHONPATH=/home/ubuntu/project/src:/opt/cool/python_3.12/site-packages/; "
                   "ln -s /opt/cool/venvs/python_3.12 /home/ubuntu/proof-env"
                   if cool else
                   "python3 -m venv /home/ubuntu/proof-env; /home/ubuntu/proof-env/bin/pip install /home/ubuntu/project")
        remote = "set -eu; cd /home/ubuntu; tar xzf worker-input.tar.gz; " + install
        remote += "; /home/ubuntu/proof-env/bin/python -c " + shlex.quote(
            "import cv2; print(cv2.__version__); print(cv2.__file__); print(cv2.getBuildInformation())")
        if cool:
            remote += "; /home/ubuntu/proof-env/bin/python -c " + shlex.quote(
                "import cv2; assert '/opt/cool' in cv2.__file__.lower() or 'cool' in cv2.getBuildInformation().lower(), 'COOL runtime not established; refusing mislabeled benchmark'")
        remote += "; mkdir -p /home/ubuntu/results"
        for label, extra in [("baseline", ""), ("selected", " --config /home/ubuntu/config.json")]:
            remote += ("; /home/ubuntu/proof-env/bin/python -m opencv_deploy_proof.cli evaluate --manifest /home/ubuntu/data/manifest.json"
                       " --split test --repeats 15 --out /home/ubuntu/results/" + label + ".json" + extra)
        result = subprocess.run(connection + [remote], timeout=1200, text=True, capture_output=True)
        (args.out / "worker.log").write_text(result.stdout + result.stderr)
        result.check_returncode()
        execute(scp + ["-r", f"ubuntu@{address}:/home/ubuntu/results", str(args.out)])
        resource["measurement_status"] = "collected"
    finally:
        if instance_id:
            ec2.terminate_instances(InstanceIds=[instance_id])
            resource["termination_requested"] = True
            # Retain IDs if deletion is not yet complete; cleanup errors must be visible.
        if imported:
            ec2.delete_key_pair(KeyName=token)
        key.unlink(missing_ok=True)
        key.with_suffix(".pub").unlink(missing_ok=True)
        if group_id:
            for _ in range(12):
                try:
                    ec2.delete_security_group(GroupId=group_id)
                    resource["security_group_deleted"] = True
                    break
                except ec2.exceptions.ClientError as error:
                    if error.response["Error"]["Code"] != "DependencyViolation":
                        raise
                    time.sleep(5)
            else:
                resource["cleanup_pending_security_group"] = group_id
        (args.out / "resources.json").write_text(json.dumps(resource, indent=2))
    print(json.dumps(resource, indent=2))


if __name__ == "__main__":
    main()
