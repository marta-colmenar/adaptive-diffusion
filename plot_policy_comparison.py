"""
plot_policy_comparison.py
==========================
Publication-ready figures comparing acquisition policies.

Figures produced
----------------
  1. per_k_curves_{metric}.pdf    – RMSE / MAE / MD-MAE / sLV-MAE vs k
                                     one line per policy, shaded CI
  2. severity_bars_{metric}.pdf   – per-severity grouped bar chart at a
                                     fixed budget k (default: 20)
  3. pointwise_heatmap_{policy}_k{k}.pdf – per-location MAE mapped to 24-2 grid
  4. compute_cost.pdf             – cumulative wall-clock time vs k
  5. summary_table.csv            – numeric summary for paper tables

Usage
-----
python plot_policy_comparison.py \
    --metrics_dir /storage/homefs/mc21s017/results/metrics/ \
    --dataset bern \
    --output_dir /storage/homefs/mc21s017/results/figures/ \
    --budget_k 20
"""

import argparse
import os
import glob

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Style
# ─────────────────────────────────────────────────────────────────────────────

POLICY_STYLE = {
    # name in metrics file  : (display label, color, linestyle, marker)
    "random":       ("A0 Random",           "#6b6b6b", "--",  "o"),
    "growing":      ("A5 Growing (clinical)","#2196F3", "-.",  "s"),
    "pop_order":    ("A6 Pop-optimised",     "#FF9800", ":",   "^"),
    "max_variance": ("A1 Max Variance",      "#4CAF50", "-",   "D"),
    "max_entropy":  ("A2 Max Entropy",       "#9C27B0", "-",   "v"),
    "mse_oracle":   ("MSE Oracle (UB)",      "#F44336", "--",  "*"),
}

SEVERITY_ORDER   = ["normal", "early", "moderate", "advanced", "severe"]
SEVERITY_LABELS  = {
    "normal":   "Normal\n(MD>−6)",
    "early":    "Early\n(−12–−6)",
    "moderate": "Moderate\n(−20–−12)",
    "advanced": "Advanced\n(−28–−20)",
    "severe":   "Severe\n(≤−28)",
}
SEVERITY_COLORS  = ["#4CAF50", "#8BC34A", "#FFC107", "#FF5722", "#9C27B0"]

METRIC_LABELS = {
    "rmse":    "RMSE (dB)",
    "mae":     "MAE (dB)",
    "md_mae":  "MD MAE (dB)",
    "slv_mae": "sLV MAE (dB)",
}

plt.rcParams.update({
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":    "tight",
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "lines.linewidth":  1.5,
})


# ─────────────────────────────────────────────────────────────────────────────
# Load all metrics files in a directory
# ─────────────────────────────────────────────────────────────────────────────

def load_all_metrics(metrics_dir, dataset):
    files   = sorted(glob.glob(os.path.join(metrics_dir, f"*{dataset}*_metrics.npz")))
    records = {}
    for f in files:
        d      = np.load(f, allow_pickle=True)
        policy = str(d["policy"])
        records[policy] = d
        print(f"  Loaded: {policy} ({os.path.basename(f)})")
    return records


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1 – per-k metric curves
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_k_curves(records, output_dir, metrics=("rmse", "mae", "md_mae", "slv_mae")):
    for metric in metrics:
        fig, ax = plt.subplots(figsize=(6, 4))
        for policy, d in records.items():
            if metric not in d:
                continue
            style  = POLICY_STYLE.get(policy, (policy, "black", "-", "o"))
            label, color, ls, marker = style
            ks   = d["prefix_ks"]
            vals = d[metric]
            ax.plot(ks, vals, color=color, linestyle=ls, label=label,
                    marker=marker, markevery=max(1, len(ks)//10), markersize=4)

        ax.set_xlabel("Number of observed locations (k)")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"Policy comparison — {METRIC_LABELS.get(metric, metric)}")
        ax.legend(loc="upper right", framealpha=0.9)
        ax.xaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, which="major", alpha=0.25)

        out = os.path.join(output_dir, f"per_k_curves_{metric}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2 – per-severity grouped bars at fixed budget k
# ─────────────────────────────────────────────────────────────────────────────

def plot_severity_bars(records, output_dir, budget_k=20,
                       metrics=("mae", "rmse")):
    for metric in metrics:
        # collect values: {policy: {severity: scalar}}
        data = {}
        for policy, d in records.items():
            data[policy] = {}
            for sname in SEVERITY_ORDER:
                key = f"sev_{sname}_{metric}"
                if key not in d:
                    continue
                arr = d[key]
                if budget_k < len(arr):
                    data[policy][sname] = float(arr[budget_k])

        policies  = [p for p in POLICY_STYLE if p in data]
        severities = [s for s in SEVERITY_ORDER
                      if any(s in data[p] for p in policies)]

        x       = np.arange(len(severities))
        n_pol   = len(policies)
        width   = 0.8 / max(n_pol, 1)

        fig, ax = plt.subplots(figsize=(8, 4))
        for pi, policy in enumerate(policies):
            style  = POLICY_STYLE.get(policy, (policy, "black", "-", "o"))
            label, color = style[0], style[1]
            vals = [data[policy].get(s, np.nan) for s in severities]
            offset = (pi - (n_pol - 1) / 2) * width
            ax.bar(x + offset, vals, width=width * 0.9,
                   color=color, label=label, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels([SEVERITY_LABELS.get(s, s) for s in severities],
                           ha="center")
        ax.set_ylabel(METRIC_LABELS.get(metric, metric))
        ax.set_title(f"Per-severity at k={budget_k} — {METRIC_LABELS.get(metric, metric)}")
        ax.legend(loc="upper left", framealpha=0.9, ncol=2)
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator(2))
        ax.grid(True, axis="y", which="major", alpha=0.25)

        out = os.path.join(output_dir, f"severity_bars_{metric}_k{budget_k}.pdf")
        fig.savefig(out)
        plt.close(fig)
        print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3 – pointwise heatmap on 24-2 grid
# ─────────────────────────────────────────────────────────────────────────────

# 24-2 Humphrey grid coordinates (degrees visual angle)
# bern has 59 points; wa (after removing 2 blind-spot cols) has 52 points
# We use a standard 10x6 layout with the blind spot removed.
# If xy coords are available we use them; otherwise we fall back to a linear index.

def _default_grid_positions(L):
    """Return (x_arr, y_arr) as placeholder linear positions."""
    n_cols = int(np.ceil(np.sqrt(L)))
    xs = np.arange(L) % n_cols
    ys = np.arange(L) // n_cols
    return xs.astype(float), ys.astype(float)


def plot_pointwise_heatmap(records, output_dir, ks_to_plot=(0, 10, 20, 30),
                           dataset="bern", data_path=""):
    """Plot per-location MAE as a scatter heatmap on the VF grid."""
    # Try to load xy coords
    xy_x, xy_y = None, None
    if data_path:
        try:
            if dataset == "wa":
                import pandas as pd
                df_xy = pd.read_csv(os.path.join(data_path, "p24d2.csv"))
                xy_x  = df_xy["x"].values.astype(float)
                xy_y  = df_xy["y"].values.astype(float)
                # remove blind-spot cols 25,34 (0-based)
                xy_x  = np.delete(xy_x, [25, 34])
                xy_y  = np.delete(xy_y, [25, 34])
            elif dataset == "bern":
                import pandas as pd
                df_xy = pd.read_excel(os.path.join(data_path, "x_y.xlsx"))
                xy_x  = df_xy.iloc[:, 0].values.astype(float)
                xy_y  = df_xy.iloc[:, 1].values.astype(float)
        except Exception as e:
            print(f"  [WARN] Could not load xy coords: {e}. Using linear layout.")

    for policy, d in records.items():
        style  = POLICY_STYLE.get(policy, (policy, "black", "-", "o"))
        label  = style[0]
        L      = int(d["L"])

        if xy_x is None or len(xy_x) != L:
            xs, ys = _default_grid_positions(L)
        else:
            xs, ys = xy_x, xy_y

        for k in ks_to_plot:
            key = f"pw_k{k:03d}_mae"
            if key not in d:
                continue
            mae_pw = d[key]  # (L,)

            fig, ax = plt.subplots(figsize=(5, 4))
            sc = ax.scatter(xs, ys, c=mae_pw, cmap="RdYlGn_r",
                            s=120, edgecolors="k", linewidths=0.3,
                            vmin=0, vmax=np.percentile(mae_pw, 95))
            cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
            cbar.set_label("MAE (dB)")
            ax.set_xlabel("x (degrees)")
            ax.set_ylabel("y (degrees)")
            ax.invert_xaxis()
            ax.set_title(f"{label}  —  pointwise MAE at k={k}")
            ax.set_aspect("equal")

            out = os.path.join(output_dir,
                               f"pointwise_heatmap_{policy}_k{k:03d}.pdf")
            fig.savefig(out)
            plt.close(fig)
            print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4 – cumulative compute cost
# ─────────────────────────────────────────────────────────────────────────────

def plot_compute_cost(records, output_dir):
    fig, ax = plt.subplots(figsize=(6, 4))
    has_data = False
    for policy, d in records.items():
        key = "timing_per_step_cumulative"
        if key not in d:
            continue
        style = POLICY_STYLE.get(policy, (policy, "black", "-", "o"))
        label, color, ls, marker = style
        ks   = d["prefix_ks"]
        vals = d[key]
        ax.plot(ks, vals, color=color, linestyle=ls, label=label,
                marker=marker, markevery=max(1, len(ks)//10), markersize=4)
        has_data = True

    if not has_data:
        plt.close(fig)
        print("  [INFO] No timing data found. Skipping compute_cost figure.")
        return

    ax.set_xlabel("Number of observed locations (k)")
    ax.set_ylabel("Cumulative time per sample (s)")
    ax.set_title("Computational cost — cumulative sampling time")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, which="major", alpha=0.25)
    out = os.path.join(output_dir, "compute_cost.pdf")
    fig.savefig(out)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Table – numeric summary
# ─────────────────────────────────────────────────────────────────────────────

def save_summary_table(records, output_dir, budget_ks=(5, 10, 20, 30)):
    rows = []
    for policy, d in records.items():
        style = POLICY_STYLE.get(policy, (policy, "black", "-", "o"))
        label = style[0]
        for k in budget_ks:
            if k >= int(d["n_prefix"]):
                continue
            row = {"Policy": label, "k": k}
            for metric in ("rmse", "mae", "md_mae", "slv_mae"):
                if metric in d:
                    row[METRIC_LABELS.get(metric, metric)] = round(float(d[metric][k]), 3)
            rows.append(row)

    df  = pd.DataFrame(rows)
    out = os.path.join(output_dir, "summary_table.csv")
    df.to_csv(out, index=False)
    print(f"  Saved: {out}")
    print(df.to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate publication-ready figures comparing acquisition policies."
    )
    parser.add_argument("--metrics_dir", type=str, required=True,
                        help="Directory containing *_metrics.npz files.")
    parser.add_argument("--dataset",   type=str, required=True, choices=["wa", "bern"])
    parser.add_argument("--data_path", type=str,
                        default="/storage/homefs/mc21s017/data_vf/",
                        help="Path to VF data (for xy coordinates).")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--budget_k",   type=int, default=20,
                        help="Fixed budget k for severity bar charts.")
    parser.add_argument("--pointwise_ks", type=int, nargs="*",
                        default=[0, 10, 20, 30],
                        help="k values for pointwise heatmaps.")
    parser.add_argument("--summary_ks", type=int, nargs="*",
                        default=[5, 10, 20, 30],
                        help="k values in the summary table.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\nLoading metrics from {args.metrics_dir} ...")
    records = load_all_metrics(args.metrics_dir, args.dataset)
    if not records:
        raise RuntimeError(f"No *{args.dataset}*_metrics.npz files found in {args.metrics_dir}")

    print(f"\nFigure 1: per-k curves")
    plot_per_k_curves(records, args.output_dir)

    print(f"\nFigure 2: severity bars at k={args.budget_k}")
    plot_severity_bars(records, args.output_dir, budget_k=args.budget_k)

    print(f"\nFigure 3: pointwise heatmaps")
    plot_pointwise_heatmap(records, args.output_dir,
                           ks_to_plot=args.pointwise_ks,
                           dataset=args.dataset, data_path=args.data_path)

    print(f"\nFigure 4: compute cost")
    plot_compute_cost(records, args.output_dir)

    print(f"\nTable: summary CSV")
    save_summary_table(records, args.output_dir, budget_ks=args.summary_ks)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
