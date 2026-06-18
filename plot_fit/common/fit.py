from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import lmfit
import numpy as np
import pandas as pd

LOSS_COL = "tail_smoothed_val_loss"


def quadratic(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * x * x + b * x + c


def linear(x: np.ndarray, slope: float, intercept: float) -> np.ndarray:
    return slope * x + intercept


@dataclass
class QuadraticOptimum:
    status: str
    opt_x: float
    opt_y: float
    nearest_depth: int
    vertex_within: bool
    report: str
    residuals: pd.DataFrame


def fit_quadratic_optimum(part: pd.DataFrame, x_col: str, y_col: str = LOSS_COL) -> QuadraticOptimum:
    part = part.sort_values(x_col).copy()
    raw = part.loc[part[y_col].idxmin()]
    raw_x = float(raw[x_col])
    raw_y = float(raw[y_col])
    if len(part) < 3 or part[x_col].nunique() < 3:
        return QuadraticOptimum("insufficient_points", raw_x, raw_y, int(raw["depth"]), False, "insufficient points\n", pd.DataFrame())
    x = np.log10(part[x_col].to_numpy(float))
    y = part[y_col].to_numpy(float)
    model = lmfit.Model(quadratic, independent_vars=["x"])
    # np.polyfit gives stable initials for the lmfit least-squares solve.
    pa, pb, pc = np.polyfit(x, y, 2)
    params = model.make_params(a=pa, b=pb, c=pc)
    result = model.fit(y, params, x=x)
    a = float(result.params["a"].value)
    b = float(result.params["b"].value)
    c = float(result.params["c"].value)
    if a <= 0:
        status = "nonconvex_raw_min"
        opt_x = raw_x
        opt_y = raw_y
        nearest_depth = int(raw["depth"])
        within = False
    else:
        x0 = -b / (2.0 * a)
        opt_x = float(10 ** x0)
        opt_y = float(quadratic(np.array([x0]), a, b, c)[0])
        nearest = part.iloc[(part[x_col] - opt_x).abs().argmin()]
        nearest_depth = int(nearest["depth"])
        within = bool(part[x_col].min() <= opt_x <= part[x_col].max())
        status = "quadratic_fit_observed" if within else "quadratic_fit_extrapolated"
    residuals = part[["run_id", "source", "depth", "flops_budget", x_col, y_col]].copy()
    residuals["log10_x"] = x
    residuals["fit_y"] = result.best_fit
    residuals["residual"] = y - result.best_fit
    return QuadraticOptimum(status, opt_x, opt_y, nearest_depth, within, result.fit_report(), residuals)


def fit_powerlaw(frontier: pd.DataFrame, x_col: str, y_col: str) -> tuple[float, float, str]:
    part = frontier.dropna(subset=[x_col, y_col]).sort_values(x_col)
    if len(part) < 2:
        return float("nan"), float("nan"), "insufficient points\n"
    x = np.log10(part[x_col].to_numpy(float))
    y = np.log10(part[y_col].to_numpy(float))
    model = lmfit.Model(linear, independent_vars=["x"])
    pslope, pintercept = np.polyfit(x, y, 1)
    params = model.make_params(slope=pslope, intercept=pintercept)
    result = model.fit(y, params, x=x)
    return float(result.params["slope"].value), float(result.params["intercept"].value), result.fit_report()


def build_frontier(df: pd.DataFrame, group_cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    rows = []
    residual_frames = []
    report_parts = []
    for keys, part in df.groupby(group_cols):
        if not isinstance(keys, tuple):
            keys = (keys,)
        group_info = dict(zip(group_cols, keys))
        part = part.sort_values("params_total")
        raw = part.loc[part[LOSS_COL].idxmin()]
        n_fit = fit_quadratic_optimum(part, "params_total")
        d_fit = fit_quadratic_optimum(part, "tokens_seen")
        row = {
            **group_info,
            "num_points": len(part),
            "status": n_fit.status,
            "opt_params": n_fit.opt_x,
            "opt_loss": n_fit.opt_y,
            "nearest_depth": n_fit.nearest_depth,
            "vertex_within_observed_range": n_fit.vertex_within,
            "d_fit_status": d_fit.status,
            "opt_tokens_direct": d_fit.opt_x,
            "opt_loss_direct_d": d_fit.opt_y,
            "nearest_depth_d": d_fit.nearest_depth,
            "d_vertex_within_observed_range": d_fit.vertex_within,
            "opt_tokens_from_constraint": float(part["flops_budget"].iloc[0]) / (6.0 * n_fit.opt_x),
            "raw_best_depth": int(raw["depth"]),
            "raw_best_params": int(raw["params_total"]),
            "raw_best_tokens": int(raw["tokens_seen"]),
            "raw_best_loss": float(raw[LOSS_COL]),
            "raw_best_on_edge": bool(raw["params_total"] in [part["params_total"].min(), part["params_total"].max()]),
        }
        rows.append(row)
        if not n_fit.residuals.empty:
            frame = n_fit.residuals.copy()
            for k, v in group_info.items():
                frame[k] = v
            frame["fit_axis"] = "params_total"
            residual_frames.append(frame)
        if not d_fit.residuals.empty:
            frame = d_fit.residuals.copy()
            for k, v in group_info.items():
                frame[k] = v
            frame["fit_axis"] = "tokens_seen"
            residual_frames.append(frame)
        group_label = ", ".join(f"{k}={v}" for k, v in group_info.items())
        report_parts.append(f"## {group_label}: L(log10 N)\n\n```text\n{n_fit.report}\n```\n")
        report_parts.append(f"## {group_label}: L(log10 D)\n\n```text\n{d_fit.report}\n```\n")
    frontier = pd.DataFrame(rows).sort_values(group_cols)
    residuals = pd.concat(residual_frames, ignore_index=True) if residual_frames else pd.DataFrame()
    return frontier, residuals, "\n".join(report_parts)


def scaling_exponents(frontier: pd.DataFrame, source_col: str | None = None) -> tuple[pd.DataFrame, str]:
    rows = []
    reports = []
    groups = frontier.groupby(source_col) if source_col else [("mixed", frontier)]
    for source, part in groups:
        obs_n = part[part["status"] == "quadratic_fit_observed"].sort_values("flops_budget")
        obs_d = part[part["d_fit_status"] == "quadratic_fit_observed"].sort_values("flops_budget")
        alpha, n_intercept, n_report = fit_powerlaw(obs_n, "flops_budget", "opt_params")
        beta, d_intercept, d_report = fit_powerlaw(obs_d, "flops_budget", "opt_tokens_direct")
        rows.append({
            "source": source,
            "fit_name": "observed_quadratic_only",
            "num_budgets_n": len(obs_n),
            "num_budgets_d": len(obs_d),
            "fit_status": "ok" if len(obs_n) >= 3 and len(obs_d) >= 3 else "weak_or_insufficient",
            "alpha_for_N_opt": alpha,
            "beta_for_D_opt_direct": beta,
            "alpha_plus_beta_direct": alpha + beta if np.isfinite(alpha) and np.isfinite(beta) else float("nan"),
            "n_intercept": n_intercept,
            "d_intercept": d_intercept,
            "used_budgets_n": ";".join(f"{v:.0e}" for v in obs_n["flops_budget"]),
            "used_budgets_d": ";".join(f"{v:.0e}" for v in obs_d["flops_budget"]),
            "non_observed_budgets_n": ";".join(f"{v:.0e}" for v in part.loc[part["status"] != "quadratic_fit_observed", "flops_budget"]),
            "non_observed_budgets_d": ";".join(f"{v:.0e}" for v in part.loc[part["d_fit_status"] != "quadratic_fit_observed", "flops_budget"]),
        })
        reports.append(f"## Power law: {source} N_opt vs C\n\n```text\n{n_report}\n```\n")
        reports.append(f"## Power law: {source} D_opt vs C\n\n```text\n{d_report}\n```\n")
    return pd.DataFrame(rows), "\n".join(reports)
