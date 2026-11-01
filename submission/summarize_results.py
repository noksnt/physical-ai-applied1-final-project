"""成功率の表とグラフを作成する"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
from paths import CSV_DIR, ROOT

FRICTIONS = (2.0, 1.0, 0.1, 0.01)
SEEDS = range(67, 167)
CUBE_FRICTION = 0.1
OUTPUT_DIR = ROOT / "results" / "submission_summary"


def read_success(path: Path) -> bool:
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1:
        raise ValueError(f"CSVに1試行分のデータがありません: {path}")
    value = rows[0]["success"].strip().lower()
    if value not in ("true", "false"):
        raise ValueError(f"successの値が不正です: {path} ({value!r})")
    return value == "true"


def main():
    summary = []
    missing = []

    for friction in FRICTIONS:
        successes = 0
        count = 0
        for seed in SEEDS:
            filename = f"gripper_{friction:g}_cube_{CUBE_FRICTION:g}_seed_{seed}.csv"
            path = CSV_DIR / filename
            if not path.is_file():
                missing.append(str(path))
                continue
            successes += int(read_success(path))
            count += 1

        if count:
            summary.append(
                {
                    "gripper_friction": friction,
                    "cube_friction": CUBE_FRICTION,
                    "trials": count,
                    "successes": successes,
                    "failures": count - successes,
                    "success_rate_percent": 100 * successes / count,
                }
            )

    if missing:
        print(f"不足CSV: {len(missing)}件（最初の10件）")
        for path in missing[:10]:
            print("  ", path)
        raise SystemExit("全400試行が揃っていないため、集計結果は保存しません。")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUTPUT_DIR / "summary.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)

    print("Gripper摩擦 | 成功数 / 試行数 | 成功率")
    for row in summary:
        print(
            f"{row['gripper_friction']:>11g} | {row['successes']:>3} / {row['trials']} | {row['success_rate_percent']:.1f}%"
        )

    # 論文図のスタイル（plotsetting.py / plotutils.py を参考にした設定）
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 9,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, ax = plt.subplots(figsize=(4, 2.5), dpi=300)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=8)

    x = list(range(len(summary)))
    rates = [row["success_rate_percent"] for row in summary]
    ax.bar(x, rates, width=0.7, color="0.7", edgecolor="black", linewidth=0.8)
    ax.set_xticks(x, [f"{row['gripper_friction']:g}" for row in summary])
    ax.set_xlabel("Gripper friction coefficient")
    ax.set_ylabel("Success rate (%)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    fig.tight_layout()
    for extension in ("png", "pdf"):
        path = OUTPUT_DIR / f"success_rates.{extension}"
        fig.savefig(path, bbox_inches="tight", pad_inches=0.05, dpi=300)
        print(f"Saved: {path}")
    plt.close(fig)
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
