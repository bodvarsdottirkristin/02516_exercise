# plot_results.py (simple)
import csv
from pathlib import Path
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

RUNS_DIR = Path("runs")

# losses.csv header: epoch,train_loss,test_loss,train_acc,test_acc
def read_csv(path: Path):
    rows = []
    with open(path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rows.append({
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "test_loss": float(row["test_loss"]),
                    "train_acc": float(row.get("train_acc", 0.0) or 0.0),
                    "test_acc": float(row.get("test_acc", 0.0) or 0.0),
                })
            except Exception:
                continue
    return rows

def parse_name_basic(name: str):
    # assume: <model>_<loss>_... (you said they're correct)
    parts = name.split("_")
    model = parts[0] if len(parts) > 0 else "unknown"
    loss  = parts[1] if len(parts) > 1 else "unknown"
    return model, loss

def main():
    run_dirs = [p for p in RUNS_DIR.glob("*") if p.is_dir()]
    if not run_dirs:
        print("No runs found in ./runs")
        return

    groups = defaultdict(list)  # model -> list of (loss, run_dir, csv_path, label_suffix)
    for rd in sorted(run_dirs):
        csv_path = rd / "losses.csv"
        if not csv_path.exists():
            continue
        model, loss = parse_name_basic(rd.name)
        # label suffix (everything after loss) just for nicer legend
        suffix = rd.name.split("_", 2)[2] if len(rd.name.split("_", 2)) == 3 else ""
        groups[model].append((loss, rd, csv_path, suffix))

    if not groups:
        print("No valid losses.csv files found.")
        return

    summary_rows = []
    empty_runs = []

    for model, items in sorted(groups.items()):
        # ---- TEST LOSS ----
        plt.figure(figsize=(8,5))
        plotted_any = False
        for loss, rd, csv_path, suffix in sorted(items, key=lambda x: x[0]):
            rows = read_csv(csv_path)
            if not rows:
                empty_runs.append(rd.name); continue
            epochs = [r["epoch"] for r in rows]
            test_loss = [r["test_loss"] for r in rows]
            if not test_loss:
                empty_runs.append(rd.name); continue
            label = f"{loss} ({suffix})" if suffix else loss
            plt.plot(epochs, test_loss, label=label)
            plotted_any = True

            best_idx = int(np.nanargmin(test_loss))
            summary_rows.append({
                "model": model,
                "loss": loss,
                "run_dir": rd.name,
                "best_epoch": epochs[best_idx],
                "best_test_loss": float(test_loss[best_idx]),
                "final_test_loss": float(test_loss[-1]),
                "final_test_acc": float(rows[-1]["test_acc"]),
            })

        if plotted_any:
            plt.title(f"{model} – Test loss per epoch")
            plt.xlabel("Epoch"); plt.ylabel("Test loss")
            plt.grid(True, alpha=0.3); plt.legend()
            out_loss = RUNS_DIR / f"{model}_test_loss.png"
            plt.tight_layout(); plt.savefig(out_loss, dpi=150)
            print(f"Saved {out_loss}")
        else:
            plt.close()
            print(f"No valid test loss curves for model '{model}'.")

        # ---- TEST ACC ----
        plt.figure(figsize=(8,5))
        plotted_any = False
        for loss, rd, csv_path, suffix in sorted(items, key=lambda x: x[0]):
            rows = read_csv(csv_path)
            if not rows: continue
            epochs = [r["epoch"] for r in rows]
            test_acc = [r["test_acc"] for r in rows]
            if not test_acc: continue
            label = f"{loss} ({suffix})" if suffix else loss
            plt.plot(epochs, test_acc, label=label)
            plotted_any = True

        if plotted_any:
            plt.title(f"{model} – Test accuracy per epoch")
            plt.xlabel("Epoch"); plt.ylabel("Test accuracy")
            plt.grid(True, alpha=0.3); plt.legend()
            out_acc = RUNS_DIR / f"{model}_test_acc.png"
            plt.tight_layout(); plt.savefig(out_acc, dpi=150)
            print(f"Saved {out_acc}")
        else:
            plt.close()
            print(f"No valid test acc curves for model '{model}'.")

    # summary.csv (if you want it)
    if summary_rows:
        summary_path = RUNS_DIR / "summary.csv"
        with open(summary_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "model","loss","run_dir","best_epoch","best_test_loss","final_test_loss","final_test_acc"
            ])
            w.writeheader()
            for row in sorted(summary_rows, key=lambda r:(r["model"], r["loss"], r["best_test_loss"])):
                w.writerow(row)
        print(f"Wrote summary: {summary_path}")

    if empty_runs:
        print("Skipped runs with empty CSV:", ", ".join(sorted(set(empty_runs))))

if __name__ == "__main__":
    main()
