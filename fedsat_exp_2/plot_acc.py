
import argparse, csv
import matplotlib.pyplot as plt

def read_csv(path):
    rows = []
    with open(path, "r", newline="") as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)
    return rows

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+", help="CSV logs from train.py --log_csv")
    ap.add_argument("--out", type=str, default="accuracy_vs_time.png")
    args = ap.parse_args()

    plt.figure()
    for log in args.logs:
        rows = read_csv(log)
        t = [float(r["t_min"]) for r in rows]
        acc = [100.0*float(r["acc"]) for r in rows]
        if rows:
            r = rows[-1]
            rn = r.get("run_name","")
            rn = f" ({rn})" if rn else ""
            label = f'{r["algo"]}-{r["dataset"]}-{r["model"]}{rn}'
        else:
            label = log
        plt.plot(t, acc, label=label)

    plt.xlabel("Simulated Wall-Time (minutes)")
    plt.ylabel("Test Accuracy (%)")
    plt.title("Federated Learning: Test Accuracy vs Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"Saved plot to {args.out}")

if __name__ == "__main__":
    main()
