#!/usr/bin/env python3
"""
Builds leakage-safe HDBSCAN spatial features.

Expected input:
    clustering/data/tree_data_with_split.csv

Outputs:
    clustering/data/hdbscan_features.csv
    clustering/data/hdbscan_group_summary.csv
    clustering/models/hdbscan_<group>.joblib

The fitted HDBSCAN models use training coordinates only. Dev/test observations
are projected into the fixed training cluster space with hdbscan prediction
utilities rather than refitting HDBSCAN on held-out coordinates.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd

try:
    import hdbscan
    from hdbscan import all_points_membership_vectors, membership_vector, approximate_predict
except ImportError as exc:
    raise SystemExit(
        "The Python package 'hdbscan' is required. Install with: "
        "pip install -r clustering/requirements.txt"
    ) from exc

# Parameters taken from dbopt_final borough scripts
BOROUGH_CONFIGS: Dict[str, dict] = {
    "brooklyn_queens": {
        "borough_values": ["Brooklyn", "Queens"],
        "prefix": "bq",
        "params": {
            "min_cluster_size": 198,
            "min_samples": 174,
            "cluster_selection_epsilon": 0.0002581228572756402,
            "cluster_selection_method": "leaf",
            "alpha": 1.0,
        },
    },
    "bronx": {
        "borough_values": ["Bronx"],
        "prefix": "bx",
        "params": {
            "min_cluster_size": 259,
            "min_samples": 205,
            "cluster_selection_epsilon": 8.866617453994765e-05,
            "cluster_selection_method": "leaf",
            "alpha": 1.0,
        },
    },
    "manhattan": {
        "borough_values": ["Manhattan"],
        "prefix": "mn",
        "params": {
            "min_cluster_size": 490,
            "min_samples": 461,
            "cluster_selection_epsilon": 0.0018052166700030662,
            "cluster_selection_method": "leaf",
            "alpha": 1.0,
        },
    },
    "staten_island": {
        "borough_values": ["Staten Island"],
        "prefix": "si",
        "params": {
            "min_cluster_size": 487,
            "min_samples": 426,
            "cluster_selection_epsilon": 0.0028394969099000394,
            "cluster_selection_method": "leaf",
            "alpha": 1.0,
        },
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/tree_data_with_split.csv"),
        help="CSV produced by clustering/01_export_split_data.R",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/hdbscan_features.csv"),
        help="Feature CSV to write",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=Path("data/hdbscan_group_summary.csv"),
        help="Per-group clustering summary CSV to write",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory where fitted HDBSCAN models will be saved",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of top cluster IDs/probabilities to emit per row",
    )
    parser.add_argument(
        "--algorithm",
        default="best",
        help="Passed to hdbscan.HDBSCAN; use boruvka_kdtree if you know it is faster locally",
    )
    parser.add_argument(
        "--core-dist-n-jobs",
        type=int,
        default=-1,
        help="Parallel jobs for HDBSCAN core distance calculations, if supported by the installed version",
    )
    return parser.parse_args()


def safe_entropy(prob_matrix: np.ndarray) -> np.ndarray:
    """Normalized entropy for each row of a nonnegative membership matrix."""
    if prob_matrix.shape[1] <= 1:
        return np.zeros(prob_matrix.shape[0], dtype=float)
    clipped = np.clip(prob_matrix, 0.0, 1.0)
    row_sums = clipped.sum(axis=1, keepdims=True)
    normalized = np.divide(
        clipped,
        row_sums,
        out=np.zeros_like(clipped, dtype=float),
        where=row_sums > 0,
    )
    log_p = np.zeros_like(normalized, dtype=float)
    positive = normalized > 0
    log_p[positive] = np.log(normalized[positive])
    entropy = -(normalized * log_p).sum(axis=1)
    return entropy / math.log(prob_matrix.shape[1])


def topk_from_membership(
    membership: np.ndarray,
    prefix: str,
    k: int,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Return top-k cluster feature columns plus top1/top2 helper arrays."""
    n_rows, n_clusters = membership.shape
    if n_clusters == 0:
        empty = pd.DataFrame(index=np.arange(n_rows))
        for rank in range(1, k + 1):
            empty[f"hdbscan_top{rank}_cluster"] = "no_cluster"
            empty[f"hdbscan_top{rank}_prob"] = 0.0
        return empty, np.zeros(n_rows), np.zeros(n_rows), np.zeros(n_rows)

    order = np.argsort(-membership, axis=1)[:, : min(k, n_clusters)]
    top_values = np.take_along_axis(membership, order, axis=1)

    out = pd.DataFrame(index=np.arange(n_rows))
    for rank in range(1, k + 1):
        if rank <= order.shape[1]:
            cluster_ids = [f"{prefix}_soft_{idx:04d}" for idx in order[:, rank - 1]]
            probs = top_values[:, rank - 1]
        else:
            cluster_ids = ["no_cluster"] * n_rows
            probs = np.zeros(n_rows)
        out[f"hdbscan_top{rank}_cluster"] = cluster_ids
        out[f"hdbscan_top{rank}_prob"] = probs

    top1 = top_values[:, 0] if top_values.shape[1] >= 1 else np.zeros(n_rows)
    top2 = top_values[:, 1] if top_values.shape[1] >= 2 else np.zeros(n_rows)
    margin = top1 - top2
    return out, top1, top2, margin


def build_feature_frame(
    group_rows: pd.DataFrame,
    membership: np.ndarray,
    hard_labels: np.ndarray,
    hard_strengths: np.ndarray,
    group_name: str,
    prefix: str,
    top_k: int,
) -> pd.DataFrame:
    """Build compact row-level HDBSCAN features for one group/split block."""
    n_rows = len(group_rows)
    if membership.ndim == 1:
        membership = membership.reshape(n_rows, -1)

    topk_frame, top1, top2, margin = topk_from_membership(membership, prefix, top_k)
    cluster_mass = np.clip(membership.sum(axis=1), 0.0, 1.0) if membership.shape[1] > 0 else np.zeros(n_rows)
    entropy = safe_entropy(membership) if membership.shape[1] > 0 else np.zeros(n_rows)

    hard_cluster = [
        f"{prefix}_noise" if int(label) == -1 else f"{prefix}_hard_{int(label):04d}"
        for label in hard_labels
    ]

    out = pd.DataFrame(
        {
            "row_id": group_rows["row_id"].to_numpy(),
            "split": group_rows["split"].to_numpy(),
            "hdbscan_fit_group": group_name,
            "hdbscan_hard_cluster": hard_cluster,
            "hdbscan_is_noise": (hard_labels == -1).astype(int),
            "hdbscan_hard_strength": np.nan_to_num(hard_strengths, nan=0.0),
            "hdbscan_margin_top1_top2": margin,
            "hdbscan_entropy_norm": entropy,
            "hdbscan_cluster_mass": cluster_mass,
            "hdbscan_off_cluster_mass": np.clip(1.0 - cluster_mass, 0.0, 1.0),
        }
    )
    return pd.concat([out.reset_index(drop=True), topk_frame.reset_index(drop=True)], axis=1)


def fit_group(
    data: pd.DataFrame,
    group_name: str,
    config: dict,
    top_k: int,
    algorithm: str,
    core_dist_n_jobs: int,
    model_dir: Path,
) -> Tuple[pd.DataFrame, dict]:
    group_mask = data["borough"].isin(config["borough_values"])
    group_data = data.loc[group_mask].copy()
    train_data = group_data[group_data["split"] == "train"].copy()

    if train_data.empty:
        raise ValueError(f"No training rows found for HDBSCAN group {group_name!r}")

    params = dict(config["params"])
    params.update(
        {
            "prediction_data": True,
            "algorithm": algorithm,
            "core_dist_n_jobs": core_dist_n_jobs,
        }
    )

    train_coords = train_data[["longitude", "latitude"]].to_numpy(dtype=float)
    clusterer = hdbscan.HDBSCAN(**params).fit(train_coords)

    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(clusterer, model_dir / f"hdbscan_{group_name}.joblib")

    feature_frames: List[pd.DataFrame] = []
    for split_name in ["train", "dev", "test"]:
        split_rows = group_data[group_data["split"] == split_name].copy()
        if split_rows.empty:
            continue
        coords = split_rows[["longitude", "latitude"]].to_numpy(dtype=float)
        if split_name == "train":
            memberships = all_points_membership_vectors(clusterer)
            hard_labels = clusterer.labels_.astype(int)
            hard_strengths = np.nan_to_num(clusterer.probabilities_, nan=0.0)
        else:
            memberships = membership_vector(clusterer, coords)
            hard_labels, hard_strengths = approximate_predict(clusterer, coords)
            hard_labels = hard_labels.astype(int)
            hard_strengths = np.nan_to_num(hard_strengths, nan=0.0)

        feature_frames.append(
            build_feature_frame(
                group_rows=split_rows,
                membership=np.asarray(memberships, dtype=float),
                hard_labels=np.asarray(hard_labels, dtype=int),
                hard_strengths=np.asarray(hard_strengths, dtype=float),
                group_name=group_name,
                prefix=config["prefix"],
                top_k=top_k,
            )
        )

    labels = clusterer.labels_
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    summary = {
        "group": group_name,
        "borough_values": ",".join(config["borough_values"]),
        "train_rows": int(len(train_data)),
        "total_rows": int(len(group_data)),
        "n_clusters_train": int(n_clusters),
        "noise_pct_train": float(np.mean(labels == -1)),
        **{f"param_{key}": value for key, value in config["params"].items()},
    }
    return pd.concat(feature_frames, ignore_index=True), summary


def validate_input(data: pd.DataFrame) -> None:
    required = {"row_id", "split", "borough", "longitude", "latitude"}
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Input is missing required columns: {sorted(missing)}")
    allowed_splits = {"train", "dev", "test"}
    observed = set(data["split"].dropna().unique())
    if not observed.issubset(allowed_splits):
        raise ValueError(f"Unexpected split values: {sorted(observed - allowed_splits)}")


def main() -> None:
    args = parse_args()
    data = pd.read_csv(args.input)
    validate_input(data)

    all_features: List[pd.DataFrame] = []
    summaries: List[dict] = []

    covered = pd.Series(False, index=data.index)
    for group_name, config in BOROUGH_CONFIGS.items():
        mask = data["borough"].isin(config["borough_values"])
        if not mask.any():
            print(f"Skipping {group_name}: no matching rows")
            continue
        print(f"Fitting HDBSCAN group: {group_name} ({int(mask.sum())} rows)")
        features, summary = fit_group(
            data=data,
            group_name=group_name,
            config=config,
            top_k=args.top_k,
            algorithm=args.algorithm,
            core_dist_n_jobs=args.core_dist_n_jobs,
            model_dir=args.model_dir,
        )
        all_features.append(features)
        summaries.append(summary)
        covered = covered | mask

    uncovered = data.loc[~covered, ["row_id", "borough"]]
    if not uncovered.empty:
        raise ValueError(
            "Some borough values are not covered by BOROUGH_CONFIGS: "
            f"{sorted(uncovered['borough'].unique())}"
        )

    out = pd.concat(all_features, ignore_index=True).sort_values("row_id")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    pd.DataFrame(summaries).to_csv(args.summary_output, index=False)

    print(f"Wrote {args.output} with {len(out)} rows and {out.shape[1]} columns")
    print(f"Wrote {args.summary_output}")


if __name__ == "__main__":
    main()
