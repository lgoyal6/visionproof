"""Reproduce a compute-only comparison from actual worker evidence and current AWS prices."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import boto3

from opencv_deploy_proof.evaluate import Requirements, gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pricing = boto3.client("pricing", region_name="us-east-1")
    rows, manifest = [], None
    for run in args.runs:
        resource = json.loads((run / "resources.json").read_text())
        if resource["region"] != "us-east-1":
            raise ValueError("price lookup currently supports us-east-1 only")
        baseline = json.loads((run / "results/baseline.json").read_text())
        selected = json.loads((run / "results/selected.json").read_text())
        if manifest and selected["manifest_sha256"] != manifest:
            raise ValueError("cannot compare different input manifests")
        manifest = selected["manifest_sha256"]
        filters = [{"Type": "TERM_MATCH", "Field": key, "Value": value} for key, value in {
            "instanceType": resource["instance_type"], "location": "US East (N. Virginia)",
            "operatingSystem": "Linux", "tenancy": "Shared", "preInstalledSw": "NA",
            "capacitystatus": "Used"}.items()]
        rates = set()
        for item in pricing.get_products(ServiceCode="AmazonEC2", Filters=filters, MaxResults=10)["PriceList"]:
            for term in json.loads(item).get("terms", {}).get("OnDemand", {}).values():
                for dimension in term["priceDimensions"].values():
                    if dimension["unit"] == "Hrs":
                        rates.add(float(dimension["pricePerUnit"]["USD"]))
        if len(rates) != 1:
            raise ValueError("no unique current EC2 hourly price; refusing to invent cost")
        hourly = rates.pop()
        milliseconds = selected["performance"]["median_batch_ms"]
        rows.append({"run": str(run), "target": resource["target"], "instance_type": resource["instance_type"],
                     "instance_id": resource["instance_id"], "hourly_usd": hourly,
                     "config": selected["config"], "opencv": selected["environment"]["opencv"],
                     "baseline_batch_ms": baseline["performance"]["median_batch_ms"],
                     "selected_batch_ms": milliseconds, "quality": selected["quality"],
                     "gate": gate(selected, baseline, Requirements()),
                     "steady_state_compute_usd_per_1000_images": hourly / 3600 * milliseconds /
                     selected["performance"]["image_count"],
                     "baseline_sha256": baseline["sha256"], "selected_sha256": selected["sha256"]})
    result = {"price_checked_utc": datetime.now(timezone.utc).isoformat(), "manifest_sha256": manifest,
              "scope": "Warm in-memory compute at full measured utilization; not all-in cost or an invoice. "
                       "Excludes idle/startup, EBS, networking, model/API and Marketplace software fees. "
                       "CPU/core/SMT mappings differ across instance families.", "rows": rows}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
