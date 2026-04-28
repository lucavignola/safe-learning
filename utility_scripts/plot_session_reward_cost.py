import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def _read_series(csv_path: Path) -> tuple[list[float], list[float]]:
    rewards: list[float] = []
    costs: list[float] = []

    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"reward", "cost"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            missing = required.difference(reader.fieldnames or [])
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            rewards.append(float(row["reward"]))
            costs.append(float(row["cost"]))

    if not rewards:
        raise ValueError("CSV does not contain any data rows")

    return rewards, costs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot reward (left) and cost (right) from an experiment session CSV."
    )
    parser.add_argument("csv_path", help="Path to the CSV file")
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output image path. If omitted, shows an interactive window.",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    rewards, costs = _read_series(csv_path)
    x = list(range(len(rewards)))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharex=True)

    axes[0].plot(x, rewards, color="tab:blue", linewidth=1.6)
    axes[0].set_title("Reward")
    axes[0].set_xlabel("Episode index")
    axes[0].set_ylabel("Reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(x, costs, color="tab:red", linewidth=1.6)
    axes[1].set_title("Cost")
    axes[1].set_xlabel("Episode index")
    axes[1].set_ylabel("Cost")
    axes[1].grid(alpha=0.3)

    fig.suptitle(csv_path.name)
    fig.tight_layout()

    if args.save:
        out_path = Path(args.save).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"Saved plot to: {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
