from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import scanpy as sc
import seaborn as sns


ROOT = Path("/data/user/hesy/projects/SpatialMETA")
RUN_ROOT = ROOT / "SpaDTA_718" / "runs" / "sm_downstream"
SAMPLES = ("X49_T", "248_T", "m1_FMP")
CLUSTER_KEY = "decalign_linear_clusters"
CONTRIBUTION_ST_KEY = "contribution_st_decalign_linear"
CONTRIBUTION_SM_KEY = "contribution_sm_decalign_linear"


def plot_one(sample: str) -> Path:
    input_path = RUN_ROOT / "inputs" / sample / f"{sample}_output.h5ad"
    output_dir = RUN_ROOT / "fig2a" / sample
    output_dir.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    adata = sc.read_h5ad(input_path)
    required = [CLUSTER_KEY, CONTRIBUTION_ST_KEY, CONTRIBUTION_SM_KEY]
    missing = [key for key in required if key not in adata.obs.columns]
    if missing:
        raise KeyError(f"{sample}: missing obs columns: {missing}")

    long_df = pd.concat(
        [
            adata.obs[[CLUSTER_KEY, CONTRIBUTION_ST_KEY]].rename(
                columns={CLUSTER_KEY: "cluster", CONTRIBUTION_ST_KEY: "contribution"}
            ).assign(modality="ST"),
            adata.obs[[CLUSTER_KEY, CONTRIBUTION_SM_KEY]].rename(
                columns={CLUSTER_KEY: "cluster", CONTRIBUTION_SM_KEY: "contribution"}
            ).assign(modality="SM"),
        ],
        ignore_index=True,
    )
    long_df["cluster"] = long_df["cluster"].astype(str)
    long_df["contribution"] = pd.to_numeric(long_df["contribution"], errors="coerce")
    long_df = long_df.dropna(subset=["cluster", "contribution"])
    long_df.to_csv(output_dir / f"{sample}_cluster_contribution_long.csv", index=False)

    summary = (
        long_df.groupby(["cluster", "modality"], observed=True)["contribution"]
        .agg(["mean", "median", "std", "count"])
        .reset_index()
    )
    summary.to_csv(output_dir / f"{sample}_cluster_contribution_summary.csv", index=False)

    cluster_order = sorted(long_df["cluster"].unique(), key=lambda value: (not value.isdigit(), int(value) if value.isdigit() else value))
    sns.set_theme(style="whitegrid", context="talk")
    fig, ax = plt.subplots(figsize=(13, 4.5))
    sns.violinplot(
        data=long_df,
        x="cluster",
        y="contribution",
        hue="modality",
        order=cluster_order,
        hue_order=["ST", "SM"],
        split=True,
        inner="quart",
        palette=["#2ec4b6", "#FFCC70"],
        scale="width",
        bw=0.2,
        cut=0,
        ax=ax,
    )
    ax.set_xlabel("cluster", fontsize=16)
    ax.set_ylabel("contribution", fontsize=16)
    ax.set_title(sample, fontsize=20, fontweight="bold")
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(title="modality", fontsize=12, title_fontsize=13, frameon=False, loc="upper right")
    # Keep the y-axis focused on the observed contribution range, as in the
    # paper-style plots, instead of compressing values into a fixed [0, 1] span.
    ax.margins(y=0.08)
    fig.tight_layout()
    for suffix, kwargs in (("png", {"dpi": 220}), ("pdf", {}), ("svg", {"format": "svg"})):
        fig.savefig(output_dir / f"{sample}_cluster_contribution_violin.{suffix}", bbox_inches="tight", **kwargs)
    plt.close(fig)
    print(f"[fig2a] wrote {output_dir}")
    return output_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot ST/SM modality contribution by SpaDTA cluster.")
    parser.add_argument("--samples", nargs="+", default=list(SAMPLES), choices=list(SAMPLES))
    args = parser.parse_args()
    for sample in args.samples:
        plot_one(sample)


if __name__ == "__main__":
    main()
