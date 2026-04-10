"""generate synthetic telecom dataset to data/raw."""
from pathlib import Path

import yaml

from src.data.synthetic import save_all


def main() -> None:
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    out_dir = Path(cfg["data"]["raw_dir"])
    summary = save_all(
        out_dir=out_dir,
        n_customers=cfg["data"]["customer_count"],
        months=cfg["data"]["months_of_transactions"],
        seed=cfg["data"]["random_seed"],
    )

    print(f"saved to {out_dir}")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
