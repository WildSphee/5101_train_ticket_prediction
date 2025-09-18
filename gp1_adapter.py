#!/usr/bin/env python3
"""
GP1 Adapter: take your already-engineered feature CSV and run it through a
compatible demand-modeling pipeline (aligned with your colleague's notebook),
PLUS a fully one-hot-encoded sklearn track.

Usage:
  python gp1_adapter.py --csv data.csv --out /tmp/gp1_out

Outputs (in --out):
  - cleaned.csv                         (snapshot after basic checks)
  - eda_describe.csv, eda_missingness.csv
  - statsmodels_OLS_summary.txt
  - statsmodels_Poisson_summary.txt
  - statsmodels_NegBin_summary.txt
  - sklearn_poisson_report.csv          (CV metrics)
  - sklearn_linear_report.csv           (CV metrics for log->count)
  - sklearn_feature_names.json          (one-hot feature map)
  - model_fit_stats.csv                 (AIC/BIC/dispersion for SM models)
  - cv_results.csv                      (SM vs SK metrics)
  - models/*.joblib                     (trained sklearn models)

Notes:
- If your CSV already includes engineered columns (e.g., log_price, days_to_departure),
  we will use them when present; otherwise we derive as needed.
- Categorical handling mirrors the notebook via statsmodels' C() and, in a separate
  track, a sklearn OneHotEncoder with handle_unknown='ignore'.
"""

import argparse
import os
import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf

# sklearn track (one-hot-everything)
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import PoissonRegressor, RidgeCV
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error
from joblib import dump
import scipy.sparse as sp


# ----------------------------------
# Helpers
# ----------------------------------

CORE_COLS = [
    "num_seats_total",
    "mean_net_ticket_price",
    "Dept_Date",
    "Purchase_Date",
    "Train_Number_All",
    "isNormCabin",
    "isReturn",
    "isOneway",
    "Customer_Cat",
]

# Extra helpful engineered columns if present
EXTRA_TIME_COLS = [
    "days_to_departure", "lead_time_days", "dept_month", "dept_weekday"
]


def read_csv_safely(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Parse dates when present
    for c in ["Dept_Date", "Purchase_Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    return df


def ensure_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # days_to_departure
    if "days_to_departure" not in df.columns:
        if "lead_time_days" in df.columns:
            df["days_to_departure"] = pd.to_numeric(df["lead_time_days"], errors="coerce")
        elif {"Dept_Date", "Purchase_Date"}.issubset(df.columns):
            df["days_to_departure"] = (df["Dept_Date"] - df["Purchase_Date"]).dt.days
        else:
            df["days_to_departure"] = np.nan

    # log_price (natural log)
    if "log_price" not in df.columns and "mean_net_ticket_price" in df.columns:
        df["log_price"] = np.log(pd.to_numeric(df["mean_net_ticket_price"], errors="coerce"))

    # log_num_seats_total (for OLS track)
    if "log_num_seats_total" not in df.columns and "num_seats_total" in df.columns:
        # +1 guard in case of zeros
        df["log_num_seats_total"] = np.log(pd.to_numeric(df["num_seats_total"], errors="coerce") + 1)

    # Binary flags -> integer 0/1 if present
    for c in ["isNormCabin", "isReturn", "isOneway"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype(int)

    # Minimal time parts (if Dept_Date exists)
    if "Dept_Date" in df.columns:
        df["dept_month"] = df["Dept_Date"].dt.month
        df["dept_weekday"] = df["Dept_Date"].dt.dayofweek

    return df


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    # Keep rows with core outcome & price
    keep = df.dropna(subset=["num_seats_total", "mean_net_ticket_price"]).copy()
    # Coerce numeric types
    keep = ensure_numeric(keep, ["num_seats_total", "mean_net_ticket_price", "days_to_departure"])
    # Make train code string-like for categorical safety
    if "Train_Number_All" in keep.columns:
        keep["Train_Number_All"] = keep["Train_Number_All"].astype(str)
    # Customer_Cat to string/category
    if "Customer_Cat" in keep.columns:
        keep["Customer_Cat"] = keep["Customer_Cat"].astype(str)
    return keep


# ----------------------------------
# Statsmodels track (mirrors colleague's formulas)
# ----------------------------------

def build_sm_formulas(df: pd.DataFrame):
    base_terms = [
        "log_price",
        "days_to_departure",
        "I(days_to_departure**2)",
        "isNormCabin",
        "isReturn",
        "isOneway",
    ]

    rhs = " + ".join(base_terms)

    if "Customer_Cat" in df.columns:
        rhs += " + C(Customer_Cat) + log_price:C(Customer_Cat)"

    if "Train_Number_All" in df.columns:
        n_trains = df["Train_Number_All"].nunique()
        if n_trains <= 200:
            rhs += " + C(Train_Number_All)"
        else:
            top_tr = df["Train_Number_All"].value_counts().nlargest(30).index
            df["Train_top30"] = df["Train_Number_All"].where(df["Train_Number_All"].isin(top_tr), other="Other")
            rhs += " + C(Train_top30)"

    # OLS is in logs; Count models use raw count LHS
    formula_ols = f"log_num_seats_total ~ {rhs}"
    formula_cnt = f"num_seats_total ~ {rhs}"
    return formula_ols, formula_cnt, df


def fit_statsmodels(df: pd.DataFrame, outdir: str):
    formula_ols, formula_cnt, df2 = build_sm_formulas(df)

    # Drop rows needed for OLS
    ols_df = df2.dropna(subset=["log_num_seats_total", "log_price", "days_to_departure"]).copy()
    ols_res = smf.ols(formula=formula_ols, data=ols_df).fit(cov_type="HC3")

    # Poisson / NegBin on available rows
    cnt_df = df2.dropna(subset=["num_seats_total", "log_price", "days_to_departure"]).copy()
    pois_res = smf.glm(formula=formula_cnt, data=cnt_df, family=sm.families.Poisson()).fit()
    nb_res   = smf.glm(formula=formula_cnt, data=cnt_df, family=sm.families.NegativeBinomial()).fit()

    # Save summaries
    with open(os.path.join(outdir, "statsmodels_OLS_summary.txt"), "w") as f:
        f.write(ols_res.summary().as_text())
    with open(os.path.join(outdir, "statsmodels_Poisson_summary.txt"), "w") as f:
        f.write(pois_res.summary().as_text())
    with open(os.path.join(outdir, "statsmodels_NegBin_summary.txt"), "w") as f:
        f.write(nb_res.summary().as_text())

    # Fit stats
    def fit_stats_sm(model):
        pearson_chi2 = np.sum(model.resid_pearson**2)
        dispersion = pearson_chi2 / model.df_resid
        llf = getattr(model, "llf", np.nan)
        # Null llf proxy if available
        ll_null = getattr(model, "null_deviance", np.nan)
        ll_null = (ll_null / -2) if pd.notna(ll_null) and ll_null != 0 else np.nan
        pseudo_r2 = 1 - (llf / ll_null) if pd.notna(ll_null) else np.nan
        return {
            "AIC": getattr(model, "aic", np.nan),
            "BIC": getattr(model, "bic", np.nan),
            "Dispersion": dispersion,
            "Pseudo_R2": pseudo_r2,
        }

    fit_df = pd.DataFrame(
        [fit_stats_sm(ols_res), fit_stats_sm(pois_res), fit_stats_sm(nb_res)],
        index=["OLS", "Poisson", "NegBin"],
    )
    fit_df.to_csv(os.path.join(outdir, "model_fit_stats.csv"))

    return ols_res, pois_res, nb_res


# ----------------------------------
# Sklearn track (one-hot everything)
# ----------------------------------

def make_sklearn_datasets(df: pd.DataFrame):
    df = df.copy()
    y_cnt = df["num_seats_total"].astype(float)

    # For linear regression, predict log(count+1)
    y_log = np.log(df["num_seats_total"].astype(float) + 1.0)

    # Candidate numerical & categorical columns
    num_cols = []
    cat_cols = []

    for c in df.columns:
        if c in {"num_seats_total"}:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)
        else:
            cat_cols.append(c)

    # Remove high-cardinality ID-like columns from num; keep as categorical
    # Train_Number_All, Customer_Cat should be categorical
    for c in ["Train_Number_All", "Customer_Cat"]:
        if c in num_cols:
            num_cols.remove(c)
        if c not in cat_cols and c in df.columns:
            cat_cols.append(c)

    # Ensure price and days are in numeric set
    for c in ["mean_net_ticket_price", "days_to_departure"]:
        if c in df.columns and c not in num_cols:
            num_cols.append(c)

    X = df[num_cols + cat_cols].copy()

    return X, y_cnt, y_log, num_cols, cat_cols


def build_sklearn_pipelines(num_cols, cat_cols):
    # Scale numeric features; one-hot categories as SPARSE to avoid OOM
    transformers = []
    if num_cols:
        transformers.append((
            "num",
            StandardScaler(with_mean=False),  # keeps CSR sparse compatibility
            num_cols,
        ))
    if cat_cols:
        transformers.append((
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            cat_cols,
        ))

    pre = ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        sparse_threshold=1.0,  # force sparse global matrix if any block is sparse
    )

    # Ridge on log(count+1) — stable, supports sparse via solver
    ridge_alphas = (10.0 ** np.linspace(-3, 3, 13))
    ridge_pipe = Pipeline([
        ("pre", pre),
        ("ridge", RidgeCV(alphas=ridge_alphas, store_cv_results=False)),
    ])

    # Lasso on log(count+1) — feature selection; accepts CSC/CSR
    lasso_alphas = np.logspace(-4, 1.5, 80)
    lasso_pipe = Pipeline([
        ("pre", pre),
        ("lasso", LassoCV(alphas=lasso_alphas, cv=5, random_state=42, n_jobs=-1, max_iter=20000)),
    ])

    # Poisson on counts — accepts sparse and is fast
    pois_pipe = Pipeline([
        ("pre", pre),
        ("pois", PoissonRegressor(alpha=1.0, max_iter=1000)),
    ])

    return ridge_pipe, lasso_pipe, pois_pipe, pre


def cv_report(pipe, X, y, transform_back=None, cv_splits=5):
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=1)
    maes, rmses = [], []
    for tr, te in kf.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        pipe.fit(Xtr, ytr)
        yhat = pipe.predict(Xte)
        if transform_back:
            yhat = transform_back(yhat)
        maes.append(mean_absolute_error(yte, yhat))
        rmses.append(np.sqrt(mean_squared_error(yte, yhat)))
    return float(np.mean(maes)), float(np.mean(rmses))


# ----------------------------------
# Main
# ----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Path to your feature CSV (e.g., data.csv)")
    ap.add_argument("--out", required=True, help="Output directory")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    # 1) Load & align
    df = read_csv_safely(args.csv)
    df = add_derived_columns(df)
    df = basic_clean(df)

    # Save a snapshot
    df.to_csv(os.path.join(args.out, "cleaned.csv"), index=False)

    # 2) Light EDA tables
    desc = df.describe(include="all").T
    desc.to_csv(os.path.join(args.out, "eda_describe.csv"))
    miss = df.isnull().sum().sort_values(ascending=False)
    miss.to_csv(os.path.join(args.out, "eda_missingness.csv"))

    # 3) Statsmodels track (colleague-compatible)
    ols_res, pois_res, nb_res = fit_statsmodels(df, args.out)

    # 4) Sklearn one-hot-everything track
    X, y_cnt, y_log, num_cols, cat_cols = make_sklearn_datasets(df)
    ridge_pipe, lasso_pipe, pois_pipe, pre = build_sklearn_pipelines(num_cols, cat_cols)

    # CV evaluation (predict back to counts for log models)
    def back_to_counts(v):
        return np.exp(v) - 1

    ridge_mae, ridge_rmse = cv_report(ridge_pipe, X, y_log, transform_back=back_to_counts)
    lasso_mae, lasso_rmse = cv_report(lasso_pipe, X, y_log, transform_back=back_to_counts)
    pois_mae, pois_rmse = cv_report(pois_pipe, X, y_cnt)

    # Train on full data & persist
    ridge_pipe.fit(X, y_log)
    lasso_pipe.fit(X, y_log)
    pois_pipe.fit(X, y_cnt)

    model_dir = os.path.join(args.out, "models")
    os.makedirs(model_dir, exist_ok=True)
    dump(ridge_pipe, os.path.join(model_dir, "ridge_logcount.joblib"))
    dump(lasso_pipe, os.path.join(model_dir, "lasso_logcount.joblib"))
    dump(pois_pipe, os.path.join(model_dir, "poisson_count.joblib"))

    # Save one-hot feature names
    try:
        pre_fit = pre.fit(X)
        feat_names = []
        if num_cols:
            feat_names += [f"num__{c}" for c in num_cols]
        if cat_cols:
            ohe = pre_fit.named_transformers_["cat"]
            ohe_names = list(ohe.get_feature_names_out(cat_cols))
            feat_names += [f"cat__{n}" for n in ohe_names]
        with open(os.path.join(args.out, "sklearn_feature_names.json"), "w") as f:
            json.dump(feat_names, f, indent=2)
    except Exception as e:
        with open(os.path.join(args.out, "sklearn_feature_names.json"), "w") as f:
            json.dump({"note": f"Could not extract names: {e}"}, f)

    # 5) Consolidated reports
    sk_poisson = pd.DataFrame({"Model": ["Sklearn Poisson"],
                               "MAE": [pois_mae],
                               "RMSE": [pois_rmse]})
    sk_ridge  = pd.DataFrame({"Model": ["Sklearn RIDGE (log->count)"],
                               "MAE": [ridge_mae],
                               "RMSE": [ridge_rmse]})
    sk_lasso  = pd.DataFrame({"Model": ["Sklearn LASSO (log->count)"],
                               "MAE": [lasso_mae],
                               "RMSE": [lasso_rmse]})
    sk_poisson.to_csv(os.path.join(args.out, "sklearn_poisson_report.csv"), index=False)
    sk_ridge.to_csv(os.path.join(args.out, "sklearn_ridge_report.csv"), index=False)
    sk_lasso.to_csv(os.path.join(args.out, "sklearn_lasso_report.csv"), index=False)

    # Cross-track quick table
    cv_all = pd.concat([
        pd.DataFrame({"Model": ["SM OLS (exp pred)", "SM Poisson", "SM NegBin"],
                      "MAE": [np.nan, np.nan, np.nan],
                      "RMSE": [np.nan, np.nan, np.nan]}),
        sk_ridge,
        sk_lasso,
        sk_poisson,
    ], ignore_index=True)
    cv_all.to_csv(os.path.join(args.out, "cv_results.csv"), index=False)

    print("Done. Outputs in:", args.out)


if __name__ == "__main__":
    main()
