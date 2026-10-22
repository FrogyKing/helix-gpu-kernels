import argparse
import json
from pathlib import Path

from helix.runtime.budget import PRICE_DATE, PRICE_SOURCE, estimate


def main():
    parser = argparse.ArgumentParser(
        description="Helix local experiment planning (no GPU dispatch)"
    )
    parser.add_argument("command", choices=["plan", "cost-report", "info"])
    parser.add_argument("--gpu", choices=["L4", "A10", "H100"], default="L4")
    parser.add_argument("--ledger", default="benchmarks/cost_ledger.json")
    args = parser.parse_args()
    if args.command == "info":
        from helix.runtime.metadata import hardware

        print(json.dumps(hardware(), indent=2))
    elif args.command == "plan":
        jobs = {"gpu_info": 120, "test_kernels": 900, "benchmark": 600}
        estimates = {k: estimate(args.gpu, v) for k, v in jobs.items()}
        print(
            json.dumps(
                {
                    "gpu": args.gpu,
                    "max_concurrency": 1,
                    "jobs_timeout_seconds": jobs,
                    "reserved_usd": estimates,
                    "total_usd": sum(estimates.values()),
                    "approval_required": True,
                    "price_date": PRICE_DATE,
                    "source": PRICE_SOURCE,
                },
                indent=2,
            )
        )
    else:
        path = Path(args.ledger)
        ledger = json.loads(path.read_text()) if path.exists() else {"reservations": []}
        ledger["total_reserved_usd"] = sum(r["reserved_usd"] for r in ledger["reservations"])
        ledger["note"] = (
            "Reservations retained on failure. Reconcile with Modal billing. Not a cap."
        )
        print(json.dumps(ledger, indent=2))


if __name__ == "__main__":
    main()
