#!/usr/bin/env python3
# plot_runs.py
# Overlay accuracy trajectories saved by train_v7.py.
# Legends = algorithm names; title = scenario + IID info.

import glob, os
import pandas as pd
import matplotlib.pyplot as plt

def parse_slug(fname):
    """
    Extracts algo, preset, iid info from the filename slug generated in train_v7.py
    e.g. algo=fedsat__preset=bremen__iid=0__... -> returns ('fedsat', 'bremen', False)
    """
    base = os.path.basename(fname)
    algo, preset, iid = "?", "?", "?"
    for part in base.split("__"):
        if part.startswith("algo="):
            algo = part.split("=", 1)[1]
        elif part.startswith("preset="):
            preset = part.split("=", 1)[1]
        elif part.startswith("iid="):
            val = part.split("=", 1)[1]
            iid = (val in ["1", "true", "True", "yes", "Y"])
    return algo, preset, iid

# Load all CSVs in 'runs'
files = sorted(glob.glob("runs/*.csv"))
if not files:
    print("No CSV files found in 'runs/'. Run train_v7.py first.")
    raise SystemExit

plt.figure(figsize=(8, 5))

# Variables for title context
title_preset = None
title_iid = None

for fp in files:
    algo, preset, iid = parse_slug(fp)
    if title_preset is None:
        title_preset = preset
        title_iid = iid

    df = pd.read_csv(fp, comment="#")
    label = algo.upper()  # legend = algorithm name
    plt.plot(df["x"], df["accuracy"], marker='o', linewidth=1.5, label=label)

plt.xlabel("Simulated time (minutes)")  # default X label
plt.ylabel("Test accuracy (%)")
plt.grid(True, linestyle="--", alpha=0.4)
plt.legend(fontsize=8)

# Title with preset and IIDness
iid_str = "IID" if title_iid else "Non-IID"
plt.title(f"Satellite scenario: {title_preset.capitalize()} ({iid_str} data)")
plt.tight_layout()
plt.show()
