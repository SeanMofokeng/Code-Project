"""
jse_stress_analysis.py

Replication code for:
"Predicting JSE TOP 40 Equity Market Stress Using the Logistic Regression
Approach with Bootstrap Inference for South Africa".

Pipeline
--------
1. Load raw quarterly macroeconomic and JSE TOP 40 data from
   JSE_Data_Template.xlsx (sheet 1_Raw_Data).
2. Build the dependent variable y_t (= 1 if next-quarter JSE log return < 0).
3. Construct five macroeconomic predictors and z-standardise them.
4. Estimate the logistic regression model using Iteratively Reweighted
   Least Squares (IRLS), implemented from first principles.
5. Run a nonparametric bootstrap (B = 5,000, seed = 42) for standard errors,
   percentile confidence intervals, odds ratios, marginal effects, and AUC.
6. Report discrimination (AUC, ROC), calibration (Brier score), and a
   confusion matrix at a 0.50 threshold.
7. Save Figures 1, 2, 3 and a summary CSV.

Software
--------
Python 3.11+, numpy, pandas, scipy, scikit-learn, matplotlib, openpyxl.
Tested against the column layout of JSE_Data_Template.xlsx.

Usage
-----
    python jse_stress_analysis.py --data JSE_Data_Template.xlsx --out results/

License: MIT.
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, brier_score_loss


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PREDICTORS = ["CPI_YoY", "Unemp", "IP_YoY", "Repo", "Spread"]
PREDICTOR_LABELS = {
    "CPI_YoY": "CPI inflation YoY (%)",
    "Unemp":   "Unemployment rate (%)",
    "IP_YoY":  "IP growth YoY (%)",
    "Repo":    "SARB repo rate (%)",
    "Spread":  "Yield spread 10Y - 91-day (pp)",
}
N_BOOT = 5000
SEED = 42
IRLS_MAX_ITER = 100
IRLS_TOL = 1e-8
THRESHOLD = 0.50


# ---------------------------------------------------------------------------
# Data loading and feature construction
# ---------------------------------------------------------------------------

def load_raw(path: str) -> pd.DataFrame:
    """Read the raw-data sheet and return a tidy DataFrame."""
    raw = pd.read_excel(path, sheet_name="1_Raw_Data", header=2)
    raw.columns = [
        "Quarter", "QuarterStart", "JSE_TRI", "CPI", "Unemp",
        "IP_Index", "Repo", "Yield10Y", "TBill",
    ]
    # Drop description row and any empty rows.
    raw = raw[raw["Quarter"].astype(str).str.match(r"^\d{4}\s*Q[1-4]$", na=False)]
    raw = raw.reset_index(drop=True)
    numeric = ["JSE_TRI", "CPI", "Unemp", "IP_Index", "Repo", "Yield10Y", "TBill"]
    for c in numeric:
        raw[c] = pd.to_numeric(raw[c], errors="coerce")
    return raw


def _quarter_sort_key(q: str) -> tuple[int, int]:
    parts = str(q).split("Q")
    return int(parts[0].strip()), int(parts[1].strip())


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """Construct y_t, x1..x5, and z1..z5 following the manuscript."""
    df = raw.copy()
    df["_sort"] = df["Quarter"].apply(_quarter_sort_key)
    df = df.sort_values("_sort").drop(columns=["_sort"]).reset_index(drop=True)

    df["LogRet"] = np.log(df["JSE_TRI"] / df["JSE_TRI"].shift(1))
    df["LogRet_Next"] = df["LogRet"].shift(-1)
    df["y"] = (df["LogRet_Next"] < 0).astype(int)

    df["CPI_YoY"] = 100.0 * (df["CPI"] / df["CPI"].shift(4) - 1.0)
    # The IP column can be supplied as either an index level (Stats SA P3041.2)
    # or as a quarterly growth rate. Detect which form is supplied and act
    # accordingly so the script works against either layout of the template.
    ip = df["IP_Index"]
    ip_looks_like_growth = (ip.abs().median() < 20.0) and (ip.min() < 0)
    if ip_looks_like_growth:
        df["IP_YoY"] = ip
    else:
        df["IP_YoY"] = 100.0 * (ip / ip.shift(4) - 1.0)
    df["Spread"]  = df["Yield10Y"] - df["TBill"]

    keep = ["Quarter", "y", "LogRet_Next"] + PREDICTORS
    out = df[keep].dropna(subset=["y", "LogRet_Next"] + PREDICTORS).reset_index(drop=True)

    for p in PREDICTORS:
        mu = out[p].mean()
        sd = out[p].std(ddof=1)
        out[f"z_{p}"] = (out[p] - mu) / sd

    return out


# ---------------------------------------------------------------------------
# IRLS estimator
# ---------------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass
class IRLSFit:
    beta: np.ndarray
    cov: np.ndarray
    se: np.ndarray
    iters: int
    converged: bool


def fit_irls(X: np.ndarray, y: np.ndarray,
             max_iter: int = IRLS_MAX_ITER,
             tol: float = IRLS_TOL) -> IRLSFit:
    """Fit logistic regression via Iteratively Reweighted Least Squares."""
    n, k = X.shape
    beta = np.zeros(k)
    for it in range(1, max_iter + 1):
        eta = X @ beta
        p = sigmoid(eta)
        W = p * (1.0 - p)
        # Ridge-style guard against perfect separation in resamples.
        W = np.clip(W, 1e-8, None)
        z = eta + (y - p) / W
        WX = X * W[:, None]
        H = X.T @ WX
        try:
            beta_new = np.linalg.solve(H, X.T @ (W * z))
        except np.linalg.LinAlgError:
            beta_new = np.linalg.lstsq(H, X.T @ (W * z), rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    # Final covariance from converged weights.
    p = sigmoid(X @ beta)
    W = np.clip(p * (1.0 - p), 1e-8, None)
    H = X.T @ (X * W[:, None])
    try:
        cov = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H)
    se = np.sqrt(np.maximum(np.diag(cov), 0.0))
    return IRLSFit(beta=beta, cov=cov, se=se, iters=it, converged=(it < max_iter))


# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

def bootstrap_logit(X: np.ndarray, y: np.ndarray,
                    B: int = N_BOOT, seed: int = SEED) -> Tuple[np.ndarray, np.ndarray]:
    """Nonparametric pair bootstrap of IRLS coefficients and predicted probs."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    coefs = np.zeros((B, X.shape[1]))
    probs = np.zeros((B, n))
    completed = 0
    attempt = 0
    while completed < B and attempt < 5 * B:
        attempt += 1
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        # Skip degenerate resamples (only one class).
        if yb.sum() == 0 or yb.sum() == n:
            continue
        try:
            fit = fit_irls(Xb, yb)
        except Exception:
            continue
        if not np.all(np.isfinite(fit.beta)):
            continue
        coefs[completed] = fit.beta
        probs[completed] = sigmoid(X @ fit.beta)  # apply to original X
        completed += 1
    if completed < B:
        coefs = coefs[:completed]
        probs = probs[:completed]
    return coefs, probs


# ---------------------------------------------------------------------------
# Marginal effects at the mean
# ---------------------------------------------------------------------------

def marginal_effects(beta: np.ndarray) -> np.ndarray:
    """ME_j = beta_j * p*(1 - p), with intercept-only baseline."""
    intercept = beta[0]
    p0 = sigmoid(intercept)
    return beta[1:] * p0 * (1.0 - p0)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def build_results_table(fit: IRLSFit, boot_coefs: np.ndarray) -> pd.DataFrame:
    names = ["Intercept"] + [f"z_{p}" for p in PREDICTORS]
    mle = fit.beta
    se_mle = fit.se
    se_boot = boot_coefs.std(axis=0, ddof=1)
    lo = np.percentile(boot_coefs, 2.5, axis=0)
    hi = np.percentile(boot_coefs, 97.5, axis=0)
    or_pt = np.exp(mle)
    or_lo = np.exp(lo)
    or_hi = np.exp(hi)
    ratio = np.where(se_mle > 0, se_boot / se_mle, np.nan)
    return pd.DataFrame({
        "term": names,
        "beta_MLE": mle,
        "SE_MLE": se_mle,
        "SE_Boot": se_boot,
        "SE_Ratio": ratio,
        "Boot_CI_Lo": lo,
        "Boot_CI_Hi": hi,
        "OddsRatio": or_pt,
        "OR_CI_Lo": or_lo,
        "OR_CI_Hi": or_hi,
    })


def confusion_at(threshold: float, y: np.ndarray, p: np.ndarray) -> dict:
    yhat = (p >= threshold).astype(int)
    tp = int(((yhat == 1) & (y == 1)).sum())
    tn = int(((yhat == 0) & (y == 0)).sum())
    fp = int(((yhat == 1) & (y == 0)).sum())
    fn = int(((yhat == 0) & (y == 1)).sum())
    n = len(y)
    return {
        "threshold": threshold,
        "accuracy":  (tp + tn) / n,
        "sensitivity": tp / max(tp + fn, 1),
        "specificity": tn / max(tn + fp, 1),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def figure_predicted_probability(quarters, y, p, base_rate, threshold, path):
    fig, ax = plt.subplots(figsize=(11, 4.5))
    idx = np.arange(len(p))
    stress_mask = y == 1
    ax.bar(idx[stress_mask], p[stress_mask], color="#c0392b", width=0.85, label="Stress quarter")
    ax.bar(idx[~stress_mask], p[~stress_mask], color="#7f8c8d", width=0.85, label="Non-stress quarter")
    ax.axhline(threshold, color="black", linestyle="--", linewidth=1, label=f"Threshold = {threshold:.2f}")
    ax.axhline(base_rate, color="#2980b9", linestyle=":", linewidth=1, label=f"Base rate = {base_rate:.3f}")
    step = max(1, len(quarters) // 12)
    ax.set_xticks(idx[::step])
    ax.set_xticklabels([quarters[i] for i in idx[::step]], rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Predicted P(Stress)")
    ax.set_title("Figure 1. Predicted probability of JSE TOP 40 equity market stress")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_marginal_effects(df_feat, boot_coefs, path):
    grid_n = 60
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    axes = axes.ravel()
    intercept_boot = boot_coefs[:, 0]
    for i, pname in enumerate(PREDICTORS):
        ax = axes[i]
        z_col = df_feat[f"z_{pname}"]
        raw_col = df_feat[pname]
        z_grid = np.linspace(z_col.min(), z_col.max(), grid_n)
        # Predicted probability when only the j-th predictor varies and others equal 0 (sample mean).
        slopes = boot_coefs[:, i + 1]
        probs = np.zeros((boot_coefs.shape[0], grid_n))
        for b in range(boot_coefs.shape[0]):
            probs[b] = 1.0 / (1.0 + np.exp(-(intercept_boot[b] + slopes[b] * z_grid)))
        med = np.median(probs, axis=0)
        lo = np.percentile(probs, 5, axis=0)
        hi = np.percentile(probs, 95, axis=0)
        # Map z back to raw units for x-axis.
        mu = raw_col.mean()
        sd = raw_col.std(ddof=1)
        x_grid = mu + z_grid * sd
        ax.plot(x_grid, med, color="#2c3e50", linewidth=1.6, label="Median")
        ax.fill_between(x_grid, lo, hi, color="#3498db", alpha=0.25, label="90% bootstrap CI")
        ax.set_xlabel(PREDICTOR_LABELS[pname])
        ax.set_ylabel("P(Stress)")
        ax.set_ylim(0, 1)
        if i == 0:
            ax.legend(fontsize=8)
    axes[-1].set_axis_off()
    fig.suptitle("Figure 2. Marginal effects with bootstrap uncertainty", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=200)
    plt.close(fig)


def figure_roc(y, p, auc, ci_lo, ci_hi, path):
    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#c0392b", linewidth=1.8,
            label=f"Model (AUC = {auc:.3f}, 95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")
    ax.plot([0, 1], [0, 1], color="black", linestyle="--", linewidth=1, label="Random")
    ax.set_xlabel("False positive rate (1 - specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title("Figure 3. ROC curve for JSE TOP 40 stress model")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to JSE_Data_Template.xlsx")
    parser.add_argument("--out",  default="results", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw = load_raw(args.data)
    feat = build_features(raw)

    z_cols = [f"z_{p}" for p in PREDICTORS]
    X = np.column_stack([np.ones(len(feat))] + [feat[c].values for c in z_cols])
    y = feat["y"].values.astype(int)

    fit = fit_irls(X, y)

    boot_coefs, boot_probs = bootstrap_logit(X, y, B=N_BOOT, seed=SEED)

    coef_table = build_results_table(fit, boot_coefs)
    coef_table.to_csv(os.path.join(args.out, "coefficients.csv"), index=False)

    me = marginal_effects(fit.beta)
    me_boot = np.zeros((boot_coefs.shape[0], len(PREDICTORS)))
    for b in range(boot_coefs.shape[0]):
        me_boot[b] = marginal_effects(boot_coefs[b])
    me_table = pd.DataFrame({
        "predictor": PREDICTORS,
        "ME": me,
        "ME_Boot_CI_Lo": np.percentile(me_boot, 2.5, axis=0),
        "ME_Boot_CI_Hi": np.percentile(me_boot, 97.5, axis=0),
    })
    me_table.to_csv(os.path.join(args.out, "marginal_effects.csv"), index=False)

    p_hat = sigmoid(X @ fit.beta)
    base_rate = y.mean()
    brier = brier_score_loss(y, p_hat)
    brier_base = brier_score_loss(y, np.full_like(p_hat, base_rate))
    brier_improve = 1.0 - brier / brier_base
    auc = roc_auc_score(y, p_hat)
    auc_boot = np.array([roc_auc_score(y, boot_probs[b]) for b in range(boot_probs.shape[0])])
    auc_lo, auc_hi = np.percentile(auc_boot, [2.5, 97.5])
    cm = confusion_at(THRESHOLD, y, p_hat)

    summary = {
        "n_obs": int(len(y)),
        "base_rate": float(base_rate),
        "AUC": float(auc),
        "AUC_95CI": [float(auc_lo), float(auc_hi)],
        "Brier": float(brier),
        "Brier_base": float(brier_base),
        "Brier_improvement_pct": float(100.0 * brier_improve),
        "Confusion_at_0.50": cm,
        "Bootstrap_B": N_BOOT,
        "Seed": SEED,
        "IRLS_iters": int(fit.iters),
        "IRLS_converged": bool(fit.converged),
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    figure_predicted_probability(
        feat["Quarter"].tolist(), y, p_hat, base_rate, THRESHOLD,
        os.path.join(args.out, "figure1_predicted_probability.png"),
    )
    figure_marginal_effects(
        feat, boot_coefs, os.path.join(args.out, "figure2_marginal_effects.png"),
    )
    figure_roc(
        y, p_hat, auc, auc_lo, auc_hi,
        os.path.join(args.out, "figure3_roc.png"),
    )

    # Predicted probabilities as a tidy CSV.
    pd.DataFrame({
        "Quarter": feat["Quarter"].values,
        "y": y,
        "p_hat": p_hat,
    }).to_csv(os.path.join(args.out, "predicted_probabilities.csv"), index=False)

    print("=" * 72)
    print("JSE TOP 40 Stress Model: replication run complete")
    print("=" * 72)
    print(f"N observations      : {len(y)}")
    print(f"Base rate           : {base_rate:.3f}")
    print(f"AUC                 : {auc:.4f}  (95% CI [{auc_lo:.4f}, {auc_hi:.4f}])")
    print(f"Brier score         : {brier:.4f}  (base {brier_base:.4f})")
    print(f"Brier improvement   : {100*brier_improve:.1f}%")
    tp, tn, fp, fn = cm["TP"], cm["TN"], cm["FP"], cm["FN"]
    print(f"Confusion @ 0.50    : TP={tp} TN={tn} FP={fp} FN={fn}")
    print(f"Outputs saved to    : {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
