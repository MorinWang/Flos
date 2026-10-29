from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, precision_recall_fscore_support


PRIMARY_HIGHER = "sparse_auprc"
PRIMARY_LOWER = "low_rank_relerr"


def topk_binary(scores: np.ndarray, k: int) -> np.ndarray:
    flat = np.asarray(scores).ravel()
    k = int(np.clip(k, 1, flat.size))
    idx = np.argpartition(flat, -k)[-k:]
    out = np.zeros(flat.size, dtype=bool)
    out[idx] = True
    return out.reshape(scores.shape)


def evaluate_decomposition(
    low_rank_hat: np.ndarray,
    sparse_hat: np.ndarray,
    low_rank_true: np.ndarray,
    sparse_true: np.ndarray,
    support_true: np.ndarray,
) -> dict[str, float]:
    eps = 1e-10
    low_rank_relerr = float(
        np.linalg.norm(low_rank_hat - low_rank_true, ord="fro") / max(np.linalg.norm(low_rank_true, ord="fro"), eps)
    )
    sparse_relerr = float(
        np.linalg.norm(sparse_hat - sparse_true, ord="fro") / max(np.linalg.norm(sparse_true, ord="fro"), eps)
    )
    y_true = support_true.ravel().astype(int)
    scores = np.abs(sparse_hat).ravel()
    if int(y_true.sum()) == 0:
        auprc = 0.0
    else:
        auprc = float(average_precision_score(y_true, scores))
    pred = topk_binary(np.abs(sparse_hat), int(max(1, support_true.sum()))).ravel().astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, pred, average="binary", zero_division=0
    )
    sparse_rmse_support = float(
        np.sqrt(np.mean((sparse_hat[support_true] - sparse_true[support_true]) ** 2))
        if support_true.any()
        else 0.0
    )
    bg_rmse = float(np.sqrt(np.mean((low_rank_hat - low_rank_true) ** 2)))
    corr_num = float(np.sum((low_rank_hat - low_rank_hat.mean()) * (low_rank_true - low_rank_true.mean())))
    corr_den = float(np.linalg.norm(low_rank_hat - low_rank_hat.mean()) * np.linalg.norm(low_rank_true - low_rank_true.mean()))
    low_rank_corr = corr_num / corr_den if corr_den > eps else 0.0
    return {
        "low_rank_relerr": low_rank_relerr,
        "low_rank_rmse": bg_rmse,
        "low_rank_corr": float(low_rank_corr),
        "sparse_relerr": sparse_relerr,
        "sparse_rmse_support": sparse_rmse_support,
        "sparse_auprc": auprc,
        "sparse_precision": float(precision),
        "sparse_recall": float(recall),
        "sparse_f1": float(f1),
    }


def curve_metrics(low_rank_hat: np.ndarray, low_rank_true: np.ndarray) -> dict[str, float]:
    # The preprocessing removes the per-month spatial mean. Use temporal RMS
    # energy so the curve remains informative after de-meaning.
    pred = np.sqrt(np.mean(low_rank_hat**2, axis=0))
    truth = np.sqrt(np.mean(low_rank_true**2, axis=0))
    rmse = float(np.sqrt(np.mean((pred - truth) ** 2)))
    mae = float(np.mean(np.abs(pred - truth)))
    den = float(np.std(pred) * np.std(truth))
    corr = float(np.mean((pred - pred.mean()) * (truth - truth.mean())) / den) if den > 1e-10 else 0.0
    return {"curve_rmse": rmse, "curve_mae": mae, "curve_corr": corr}


def summarize_mean_std(df: pd.DataFrame, group_cols: list[str], metric_cols: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row: dict[str, object] = {col: key for col, key in zip(group_cols, keys)}
        for metric in metric_cols:
            row[f"{metric}_mean"] = float(grp[metric].mean())
            row[f"{metric}_std"] = float(grp[metric].std(ddof=1) if len(grp) > 1 else 0.0)
        rows.append(row)
    return pd.DataFrame(rows)


def paired_ttest_against_best(
    df: pd.DataFrame,
    dataset_col: str = "dataset",
    method_col: str = "method",
    seed_col: str = "seed",
    proposed: str = "FLoS-RPCA",
    metric: str = PRIMARY_HIGHER,
    higher_is_better: bool = True,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for dataset, grp in df.groupby(dataset_col):
        means = grp.groupby(method_col)[metric].mean().sort_values(ascending=not higher_is_better)
        baselines = means.drop(index=proposed, errors="ignore")
        if baselines.empty or proposed not in means.index:
            continue
        best = str(baselines.index[0])
        pivot = grp.pivot_table(index=seed_col, columns=method_col, values=metric, aggfunc="mean")
        common = pivot[[proposed, best]].dropna()
        if len(common) < 2:
            pvalue = np.nan
            stat = np.nan
        else:
            stat, pvalue = stats.ttest_rel(common[proposed], common[best], alternative="two-sided")
        improvement = means[proposed] > means[best] if higher_is_better else means[proposed] < means[best]
        rows.append(
            {
                "dataset": dataset,
                "proposed": proposed,
                "best_baseline": best,
                "proposed_mean": float(means[proposed]),
                "best_baseline_mean": float(means[best]),
                "t_stat": float(stat) if np.isfinite(stat) else np.nan,
                "p_value": float(pvalue) if np.isfinite(pvalue) else np.nan,
                "significant_p_lt_0.05": bool(improvement and np.isfinite(pvalue) and pvalue < 0.05),
            }
        )
    return pd.DataFrame(rows)


def format_mean_std(mean: float, std: float, higher: bool = True, star: bool = False) -> str:
    text = f"{mean:.4f} +/- {std:.4f}"
    if star:
        text += " \\textbf{{\\Large *}}"
    return text
