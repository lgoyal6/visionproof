import argparse
import json
from pathlib import Path


class ArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        self.exit(3, json.dumps({'status': 'error', 'error': message}) + '\n')


def main():
    parser = ArgumentParser(description="Reproduce an OpenCV deployment qualification experiment")
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--data", type=Path, required=True)
    prepare.add_argument("--limit", type=int, default=80, help="maximum image pairs per split")
    fresh = commands.add_parser("prepare-fresh")
    fresh.add_argument("--original", type=Path, required=True)
    fresh.add_argument("--out", type=Path, required=True)
    fresh.add_argument("--limit", type=int, default=64)
    reserved = commands.add_parser('prepare-reserved', help='declare fresh images on calibration-exposed PCB boards')
    reserved.add_argument('--original', type=Path, action='append', required=True)
    reserved.add_argument('--out', type=Path, required=True)
    reserved.add_argument('--limit', type=int, default=160)
    people = commands.add_parser('prepare-pedestrians', help='fetch pinned Penn-Fudan and declare source-location splits')
    people.add_argument('--data', type=Path, required=True)
    people.add_argument('--limit', type=int, default=48)
    check = commands.add_parser('check', help='CI regression gate: 0 pass, 2 regression, 3 invalid/error')
    check.add_argument('--manifest', type=Path, required=True)
    check.add_argument('--candidate-config', type=Path, required=True)
    check.add_argument('--baseline-config', type=Path)
    check.add_argument('--out', type=Path, required=True)
    check.add_argument('--rounds', type=int, default=5)
    check.add_argument('--repeats', type=int, default=5)
    check.add_argument('--seed', type=int, default=1729)
    comparison = commands.add_parser('compare', help='randomized replication of predeclared frozen configurations')
    comparison.add_argument('--manifest', type=Path, required=True)
    comparison.add_argument('--configs', type=Path, required=True, help='JSON mapping: baseline and named configurations')
    comparison.add_argument('--out', type=Path, required=True)
    comparison.add_argument('--rounds', type=int, default=7)
    comparison.add_argument('--repeats', type=int, default=7)
    comparison.add_argument('--seed', type=int, default=1729)
    evaluate = commands.add_parser("evaluate")
    evaluate.add_argument("--manifest", type=Path, required=True)
    evaluate.add_argument("--split", choices=["calibration", "development", "test"], required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--out", type=Path, required=True)
    evaluate.add_argument("--repeats", type=int, default=9)
    proof = commands.add_parser("proof")
    proof.add_argument("--manifest", type=Path, required=True)
    proof.add_argument("--out", type=Path, required=True)
    proof.add_argument("--provider", choices=["rules", "bedrock", "anthropic"], default="bedrock")
    proof.add_argument("--model")
    proof.add_argument("--budget", type=int, default=4)
    proof.add_argument("--repeats", type=int, default=9)
    preview = commands.add_parser("serve")
    preview.add_argument("--manifest", type=Path, required=True)
    preview.add_argument("--run", type=Path, required=True)
    preview.add_argument("--port", type=int, default=8765)
    preview.add_argument("--evidence-root", type=Path)
    preview.add_argument("--model", help="optional Bedrock model ID for user-triggered qualification jobs")
    public = commands.add_parser('export-public', help='export a sanitized read-only replay')
    public.add_argument('--evidence-root', type=Path, required=True)
    public.add_argument('--out', type=Path, required=True)
    public.add_argument('--run', action='append', required=True)
    public.add_argument('--default-run', required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        from .data import prepare as fetch
        path = fetch(args.data, args.limit)
        print(path)
    elif args.command == "prepare-fresh":
        from .protocol import fresh_manifest
        print(fresh_manifest(args.original, args.out, args.limit))
    elif args.command == 'prepare-reserved':
        from .protocol import reserved_manifest
        print(reserved_manifest(args.original, args.out, args.limit))
    elif args.command == 'prepare-pedestrians':
        from .pedestrian import prepare_pedestrians
        print(prepare_pedestrians(args.data, args.limit))
    elif args.command in ('check', 'compare'):
        from .comparison import compare
        from .adapters import pipeline_for
        try:
            if args.command == 'check':
                config_type = pipeline_for(args.manifest).config_type
                baseline = json.loads(args.baseline_config.read_text()) if args.baseline_config else config_type().to_dict()
                configs = {'baseline': baseline, 'candidate': json.loads(args.candidate_config.read_text())}
            else:
                configs = json.loads(args.configs.read_text())
            result = compare(args.manifest, configs, args.out, args.rounds, args.repeats, args.seed)
        except Exception as error:
            print(json.dumps({'status': 'error', 'error': str(error)}))
            raise SystemExit(3)
        print(json.dumps({'status': 'passed' if all(row['gate']['passed'] for row in result['rows']) else 'blocked',
                          'report': str(args.out / 'summary.md'), 'sha256': result['sha256']}))
        if args.command == 'check' and not all(row['gate']['passed'] for row in result['rows']):
            raise SystemExit(2)
    elif args.command == "evaluate":
        from .evaluate import write_result
        from .adapters import pipeline_for
        pipeline = pipeline_for(args.manifest)
        config = pipeline.config_type(**json.loads(args.config.read_text())) if args.config else pipeline.config_type()
        result = pipeline.evaluator(args.manifest, args.split, args.repeats).run(config)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        write_result(args.out, result)
        print(json.dumps({"quality": result["quality"], "performance": result["performance"]}))
    elif args.command == "proof":
        from .experiment import proof as run
        print(json.dumps(run(args.manifest, args.out, args.provider, args.model, args.budget, args.repeats), indent=2))
    elif args.command == 'export-public':
        from .public_demo import export_public
        print(json.dumps(export_public(args.evidence_root, args.out, args.run, args.default_run)))
    else:
        from .serve import serve
        serve(args.manifest, args.run, port=args.port, evidence_root=args.evidence_root, model=args.model)


if __name__ == "__main__":
    main()
