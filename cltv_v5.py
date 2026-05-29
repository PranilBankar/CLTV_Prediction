"""
==============================================================================
VahanBima CLTV Prediction — v5
Target: R² > 0.30 on original CLTV scale
==============================================================================

WHAT'S NEW vs v4:
─────────────────
1. XGBOOST (DART booster) — Model D
   DART randomly drops trees during boosting, reducing the outsized influence
   of a few dominant trees that latch on to outliers. Adds high-quality
   diversity to the ensemble that neither CatBoost nor LightGBM provides.

2. EXTRATREES — Model E
   Gradient boosters reduce bias but can be unstable on high-variance targets.
   ExtraTrees reduces variance through extreme randomization (random split
   thresholds). Ridge stacking learns to lean on this when GBMs overfit.
//
3. SMARTER TARGET ENCODING (Bayesian Prior Smoothing)
   Standard mean-encoding on rare category combos is noisy — a group with
   only 2 rows has an unreliable mean. Bayesian smoothing blends each group's
   mean toward the global mean, weighted by sample count:
       smooth_mean = (n * group_mean + m * global_mean) / (n + m)
   where m is the smoothing factor (~20). This eliminates noise on rare groups
   and gives stable, reliable encodings.

4. RICHER FEATURE ENGINEERING
   • Claim deviation z-score per (income × policy_type) segment —
     "is this customer's claim 2σ above their peers?"
   • Claim deviation z-score per (area × income) segment
   • Polynomial claim features: claim², claim × vintage²
   • Group aggregated statistics: group median, p25, p75 of claim_amount
   • Ratio features: claim vs group mean/median
   • Claim amount bucket × income interaction

5. QUANTILE REGRESSION (LightGBM) — Model F
   A quantile (p=0.5) model predicts the median CLTV. This is robust to
   outliers by construction, and its OOF predictions add a different
   "signal axis" to the meta-learner that reduces stacking error.

6. MULTI-LEVEL STACKING
   Level-1: 5 base models (CatBoost, HistGB, LightGBM-RMSE,
            XGBoost-DART, ExtraTrees)
   Level-1.5: LightGBM-Quantile as additional signal
   Level-2: Ridge meta-learner with original CLTV as target
   Ridge's L2 penalty prevents over-weighting any single model.

7. DEEPER HYPERPARAMETER TUNING
   • CatBoost: depth=9, more iterations, lower LR
   • LightGBM: more leaves (255), lower colsample to fight variance
   • XGBoost: max_depth=8, min_child_weight=20, subsample=0.75

8. ADAPTIVE WINSORIZATION
   Still winsorize at 99th pct for all models EXCEPT ExtraTrees,
   which is trained on full range (it handles outliers gracefully via
   variance reduction, and gives the stacker a non-winsorized signal).

==============================================================================
QUICK START:
  Place train_data.csv and test_data.csv in the same directory as this script
  (or in a Data/ subdirectory — the script checks both).
  Then: python cltv_v5.py
==============================================================================
"""

# ─── INSTALLS (uncomment if missing) ─────────────────────────────────────────
# import subprocess, sys
# for pkg in ["catboost", "lightgbm", "xgboost"]:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, gc, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection    import StratifiedKFold, KFold
from sklearn.metrics            import r2_score
from sklearn.ensemble           import (HistGradientBoostingRegressor,
                                         ExtraTreesRegressor)
from sklearn.linear_model       import Ridge
from sklearn.preprocessing      import OrdinalEncoder

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost  as xgb

SEED   = 42
N_FOLD = 5
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)
START  = time.time()

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  CLTV v5  —  Loading Data")
print("=" * 65)

# Check both flat and Data/ subdirectory
def _load(name):
    for p in [name, os.path.join("Data", name)]:
        if os.path.exists(p):
            return pd.read_csv(p)
    raise FileNotFoundError(f"{name} not found in ./ or ./Data/")

train = _load("train_data.csv")
test  = _load("test_data.csv")
TARGET = "cltv"

print(f"Train : {train.shape}   Test : {test.shape}")
print(f"CLTV  → mean={train[TARGET].mean():.0f}  "
      f"std={train[TARGET].std():.0f}  max={train[TARGET].max():.0f}")

# ─── 2. TARGET WINSORIZATION ──────────────────────────────────────────────────
WINSOR_PCT   = 99
cltv_cap     = np.percentile(train[TARGET], WINSOR_PCT)
cltv_floor   = train[TARGET].min()
cltv_true_max = train[TARGET].max()

print(f"\nWinsorization {WINSOR_PCT}th pct cap : {cltv_cap:.0f}")
n_capped = (train[TARGET] > cltv_cap).sum()
print(f"Rows capped   : {n_capped} ({100*n_capped/len(train):.1f}%)")

train["cltv_winsor"] = np.clip(train[TARGET], cltv_floor, cltv_cap)

# ─── 3. INCOME ORDINAL MAP ────────────────────────────────────────────────────
INCOME_ORDER = {"<=2L": 1, "2L-5L": 2, "5L-10L": 3, "More than 10L": 4}
all_inc  = pd.concat([train["income"], test["income"]])
income_map = {v: INCOME_ORDER.get(v, i+1)
              for i, v in enumerate(sorted(all_inc.unique()))}
print(f"\nIncome ordinal map : {income_map}")

# ─── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
def feature_engineer(df, income_map):
    df = df.copy()

    # ── ordinal income ────────────────────────────────────────────────────────
    df["income_num"]        = df["income"].map(income_map).astype(float)

    # ── policy dummies ────────────────────────────────────────────────────────
    df["multi_policy"]      = (df["num_policies"] != "1").astype(int)

    # ── claim transformations ─────────────────────────────────────────────────
    df["claim_flag"]        = (df["claim_amount"] > 0).astype(int)
    df["zero_claim"]        = (df["claim_amount"] == 0).astype(int)
    df["log_claim"]         = np.log1p(df["claim_amount"])
    df["sqrt_claim"]        = np.sqrt(df["claim_amount"])
    df["claim_sq"]          = df["claim_amount"] ** 2           # NEW: polynomial
    df["claim_per_yr"]      = df["claim_amount"] / (df["vintage"] + 1)
    df["log_claim_per_yr"]  = np.log1p(df["claim_per_yr"])
    df["high_claim"]        = (df["claim_amount"] > 6094).astype(int)  # >75th pct

    # ── vintage features ──────────────────────────────────────────────────────
    df["vintage_sq"]        = df["vintage"] ** 2
    df["vintage_cubed"]     = df["vintage"] ** 3                # NEW: cubic
    df["log_vintage"]       = np.log1p(df["vintage"])

    # ── interaction features ──────────────────────────────────────────────────
    df["v_x_logclaim"]      = df["vintage"] * df["log_claim"]
    df["vsq_x_logclaim"]    = df["vintage_sq"] * df["log_claim"]   # NEW
    df["income_x_claim"]    = df["income_num"] * df["log_claim"]
    df["income_x_vintage"]  = df["income_num"] * df["vintage"]
    df["income_x_claimsq"]  = df["income_num"] * df["claim_sq"]    # NEW
    df["high_income"]       = (df["income_num"] >= 3).astype(int)
    df["inc_x_mpol"]        = df["income_num"] * df["multi_policy"]

    # ── income-claim ratio ────────────────────────────────────────────────────
    df["claim_per_income"]  = df["claim_amount"] / (df["income_num"] + 1)  # NEW

    # ── string interaction columns (for CatBoost native cat handling) ─────────
    df["policy_type"]   = df["policy"].astype(str) + "_" + df["type_of_policy"].astype(str)
    df["inc_policy"]    = df["income"].astype(str)  + "_" + df["policy"].astype(str)
    df["inc_type"]      = df["income"].astype(str)  + "_" + df["type_of_policy"].astype(str)
    df["area_qual"]     = df["area"].astype(str)    + "_" + df["qualification"].astype(str)
    df["inc_area"]      = df["income"].astype(str)  + "_" + df["area"].astype(str)
    df["pol_type_inc"]  = df["policy_type"].astype(str) + "_" + df["income"].astype(str)
    df["inc_vint"]      = df["income"].astype(str)  + "_" + df["vintage"].astype(str)
    df["pol_vint"]      = df["policy_type"].astype(str) + "_" + df["vintage"].astype(str)
    df["gen_mar"]       = df["gender"].astype(str)  + "_" + df["marital_status"].astype(str)
    df["area_inc_pol"]  = (df["area"].astype(str) + "_" + df["income"].astype(str)   # NEW: 3-way
                           + "_" + df["policy_type"].astype(str))
    df["qual_inc"]      = df["qualification"].astype(str) + "_" + df["income"].astype(str)

    # ── bucket features ───────────────────────────────────────────────────────
    df["vintage_bucket"] = pd.cut(df["vintage"], bins=[-1, 1, 3, 6, 100],
                                  labels=["new", "mid", "old", "very_old"]).astype(str)
    df["claim_bin"]      = pd.qcut(df["claim_amount"], q=10,
                                   labels=False, duplicates="drop").astype(str)

    return df

train = feature_engineer(train, income_map)
test  = feature_engineer(test,  income_map)

# ─── 5. CLAIM PERCENTILE RANK WITHIN SEGMENT ─────────────────────────────────
# "Is this customer's claim big or small relative to their peer group?"
n_tr = len(train)
tr_r = train.reset_index(drop=True)
te_r = test.reset_index(drop=True)
all_ = pd.concat([tr_r, te_r], axis=0, ignore_index=True)

RANK_SEGS = {
    "rank_in_income"      : ["income"],
    "rank_in_policy"      : ["policy_type"],
    "rank_in_inc_pol"     : ["income", "policy_type"],
    "rank_in_area_inc"    : ["area", "income"],           # NEW
    "rank_in_inc_area_pol": ["income", "area", "policy"],  # NEW
}
for col, grp in RANK_SEGS.items():
    all_[col] = all_.groupby(grp)["claim_amount"].rank(pct=True)
    train[col] = all_.iloc[:n_tr][col].values
    test[col]  = all_.iloc[n_tr:][col].values

RANK_COLS = list(RANK_SEGS.keys())

# ─── 6. CLAIM DEVIATION Z-SCORE PER SEGMENT (NEW) ────────────────────────────
# Tells the model: "How unusual is this customer's claim vs their segment peers?"
# A high z-score (e.g., +3) within a segment is a strong CLTV signal.
print("\n[Step] Claim deviation z-scores per segment …")

ZSCORE_SEGS = {
    "claim_z_inc_pol"    : ["income", "policy_type"],
    "claim_z_area_inc"   : ["area", "income"],
    "claim_z_inc_area_pt": ["income", "area", "policy_type"],
}
all2_ = pd.concat([train.reset_index(drop=True),
                   test.reset_index(drop=True)], axis=0, ignore_index=True)

for col, grp in ZSCORE_SEGS.items():
    seg_mean = all2_.groupby(grp)["claim_amount"].transform("mean")
    seg_std  = all2_.groupby(grp)["claim_amount"].transform("std").fillna(1)
    z        = (all2_["claim_amount"] - seg_mean) / (seg_std + 1e-6)
    train[col] = z.iloc[:n_tr].values
    test[col]  = z.iloc[n_tr:].values

ZSCORE_COLS = list(ZSCORE_SEGS.keys())

# ─── 7. GROUP AGGREGATES FOR CLAIM AMOUNT (NEW) ──────────────────────────────
# Rather than just mean/std of CLTV, also give the model percentile
# information about claim_amount per segment — cheaper than target encoding
# and leak-free because claim_amount is a feature, not the target.
print("[Step] Group aggregates for claim_amount …")

GROUP_AGG_SEGS = {
    "inc_pol_claim": ["income", "policy_type"],
    "area_inc_claim": ["area", "income"],
}
all3_ = pd.concat([train.reset_index(drop=True),
                   test.reset_index(drop=True)], axis=0, ignore_index=True)

for feat_prefix, grp in GROUP_AGG_SEGS.items():
    for stat, fn in [("med", "median"), ("p25", lambda x: x.quantile(0.25)),
                     ("p75", lambda x: x.quantile(0.75))]:
        col = f"{feat_prefix}_{stat}"
        val = all3_.groupby(grp)["claim_amount"].transform(fn)
        train[col] = val.iloc[:n_tr].values
        test[col]  = val.iloc[n_tr:].values

GROUP_AGG_COLS = [f"{p}_{s}"
                  for p in GROUP_AGG_SEGS
                  for s in ["med", "p25", "p75"]]

# Ratio: how does this customer's claim compare to group median?
for prefix in GROUP_AGG_SEGS:
    col = f"{prefix}_claim_ratio"
    train[col] = train["claim_amount"] / (train[f"{prefix}_med"] + 1)
    test[col]  = test["claim_amount"]  / (test[f"{prefix}_med"] + 1)
    GROUP_AGG_COLS.append(col)

# ─── 8. BAYESIAN SMOOTHED TARGET ENCODING (CV-SAFE) ──────────────────────────
print("[Step] Bayesian smoothed target encoding …")

TE_SEGS = [
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy",
    "policy_type", "inc_policy", "inc_type",
    "area_qual", "inc_area", "pol_type_inc", "inc_vint",
    "area_inc_pol", "qual_inc",              # NEW 3-way combos
]

def bayesian_cv_te(train_df, test_df, cols, target, n_splits=5,
                   seed=42, smoothing=20):
    """
    Bayesian (smoothed) target encoding, computed out-of-fold for train.
    smooth_mean = (n * group_mean + m * global_mean) / (n + m)
    where m = smoothing parameter. Rare groups get pulled toward global mean.
    """
    kf     = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gmean  = train_df[target].mean()
    gstd   = train_df[target].std()
    out_tr = pd.DataFrame(index=train_df.index)
    out_te = pd.DataFrame(index=test_df.index)

    for col in cols:
        mean_arr   = np.full(len(train_df), np.nan)
        std_arr    = np.full(len(train_df), np.nan)
        mean_te_f  = np.zeros((len(test_df), n_splits))
        std_te_f   = np.zeros((len(test_df), n_splits))

        for f, (tri, vali) in enumerate(kf.split(train_df)):
            tr_f  = train_df.iloc[tri]
            val_f = train_df.iloc[vali]

            # Group stats on fold's training data
            grp_agg = tr_f.groupby(col)[target].agg(["mean", "std", "count"])
            grp_agg["smooth_mean"] = (
                (grp_agg["count"] * grp_agg["mean"] + smoothing * gmean) /
                (grp_agg["count"] + smoothing)
            )
            grp_agg["smooth_std"]  = grp_agg["std"].fillna(gstd)

            mean_arr[vali]   = val_f[col].map(grp_agg["smooth_mean"]).fillna(gmean).values
            std_arr[vali]    = val_f[col].map(grp_agg["smooth_std"]).fillna(gstd).values
            mean_te_f[:, f]  = test_df[col].map(grp_agg["smooth_mean"]).fillna(gmean).values
            std_te_f[:, f]   = test_df[col].map(grp_agg["smooth_std"]).fillna(gstd).values

        out_tr[col + "_te"]     = np.where(np.isnan(mean_arr), gmean, mean_arr)
        out_tr[col + "_te_std"] = np.where(np.isnan(std_arr), gstd, std_arr)
        out_te[col + "_te"]     = mean_te_f.mean(axis=1)
        out_te[col + "_te_std"] = std_te_f.mean(axis=1)

    return out_tr, out_te

te_train, te_test = bayesian_cv_te(train, test, TE_SEGS, "cltv_winsor",
                                    N_FOLD, SEED, smoothing=20)
train = pd.concat([train, te_train], axis=1)
test  = pd.concat([test,  te_test],  axis=1)
TE_COLS = te_train.columns.tolist()
print(f"  → {len(TE_COLS)} Bayesian target-encoding columns")

# ─── 9. DEFINE FEATURE SETS ───────────────────────────────────────────────────
CAT_FEATURES = [
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy", "num_policies",
    "policy_type", "inc_policy", "inc_type",
    "area_qual", "inc_area", "pol_type_inc",
    "inc_vint", "pol_vint", "gen_mar",
    "area_inc_pol", "qual_inc",              # NEW
    "vintage_bucket", "claim_bin",
]

NUM_FEATURES = [
    "marital_status", "vintage", "vintage_sq", "vintage_cubed", "log_vintage",
    "claim_amount", "claim_sq",
    "income_num", "multi_policy",
    "claim_flag", "zero_claim", "log_claim", "sqrt_claim",
    "claim_per_yr", "log_claim_per_yr",
    "high_income", "high_claim",
    "income_x_claim", "income_x_vintage", "v_x_logclaim",
    "vsq_x_logclaim", "income_x_claimsq",  # NEW polynomials
    "inc_x_mpol", "claim_per_income",       # NEW ratios
] + RANK_COLS + ZSCORE_COLS + GROUP_AGG_COLS + TE_COLS

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
print(f"\n  CAT: {len(CAT_FEATURES)}  "
      f"NUM: {len(NUM_FEATURES)}  "
      f"TOTAL: {len(ALL_FEATURES)}")

# ─── 10. PREPARE ARRAYS ───────────────────────────────────────────────────────
X      = train[ALL_FEATURES].copy()
y_raw  = train[TARGET].astype(float)
y_win  = train["cltv_winsor"].astype(float)
X_test = test[ALL_FEATURES].copy()

for col in CAT_FEATURES:
    X[col]      = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]
y_binned = pd.qcut(np.log1p(y_raw), q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A — CatBoost (deeper, more iterations, lower LR)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL A : CatBoost (depth=9, winsorized CLTV)")
print("=" * 65)

CB_PARAMS = {
    "iterations"          : 3000,
    "learning_rate"       : 0.03,       # slower but more accurate
    "depth"               : 9,          # deeper than v4's 7
    "l2_leaf_reg"         : 3.0,
    "bagging_temperature" : 0.5,
    "random_strength"     : 1.0,
    "border_count"        : 254,        # more bins = finer splits
    "loss_function"       : "RMSE",
    "eval_metric"         : "RMSE",
    "od_type"             : "Iter",
    "od_wait"             : 200,
    "random_seed"         : SEED,
    "verbose"             : 0,
    "thread_count"        : -1,
}

oof_A  = np.zeros(len(train))
pred_A = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X, y_binned), 1):
    Xtr, Xval   = X.iloc[tr_i], X.iloc[val_i]
    ytr, yval   = y_win.iloc[tr_i], y_win.iloc[val_i]
    y_raw_val   = y_raw.iloc[val_i]

    tr_pool  = Pool(Xtr,  ytr,  cat_features=cat_idx)
    val_pool = Pool(Xval, yval, cat_features=cat_idx)

    m = CatBoostRegressor(**CB_PARAMS)
    m.fit(tr_pool, eval_set=val_pool, use_best_model=True)

    oof_A[val_i] = m.predict(Xval)
    pred_A      += m.predict(X_test) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_A[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}  "
          f"best_iter={m.best_iteration_}")
    del m, tr_pool, val_pool; gc.collect()

r2_A = r2_score(y_raw, oof_A)
print(f"\n  [Model A] OOF R² (original CLTV): {r2_A:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B — HistGradientBoosting (fast sklearn)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL B : HistGradientBoosting (sklearn)")
print("=" * 65)

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_hgb      = X.copy()
X_test_hgb = X_test.copy()
X_hgb[CAT_FEATURES]      = oe.fit_transform(X_hgb[CAT_FEATURES].astype(str))
X_test_hgb[CAT_FEATURES] = oe.transform(X_test_hgb[CAT_FEATURES].astype(str))

HGB_PARAMS = dict(
    max_iter            = 1500,
    learning_rate       = 0.04,
    max_depth           = 9,
    min_samples_leaf    = 15,
    l2_regularization   = 0.5,
    max_bins            = 255,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 60,
    random_state        = SEED,
    categorical_features= list(range(len(CAT_FEATURES))),
)

oof_B  = np.zeros(len(train))
pred_B = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X_hgb, y_binned), 1):
    Xtr, Xval  = X_hgb.iloc[tr_i], X_hgb.iloc[val_i]
    ytr        = y_win.iloc[tr_i]
    y_raw_val  = y_raw.iloc[val_i]

    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    m.fit(Xtr, ytr)

    oof_B[val_i] = m.predict(Xval)
    pred_B      += m.predict(X_test_hgb) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_B[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}  "
          f"n_iter={m.n_iter_}")
    del m; gc.collect()

r2_B = r2_score(y_raw, oof_B)
print(f"\n  [Model B] OOF R² (original CLTV): {r2_B:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL C — LightGBM RMSE on winsorized CLTV
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL C : LightGBM RMSE (winsorized CLTV)")
print("=" * 65)

X_lgb      = X.copy()
X_test_lgb = X_test.copy()
for col in CAT_FEATURES:
    X_lgb[col]      = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

LGB_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 3000,
    learning_rate     = 0.03,
    num_leaves        = 255,            # more than v4's 127
    min_child_samples = 15,
    subsample         = 0.8,
    colsample_bytree  = 0.7,            # lower = less overfit
    reg_alpha         = 0.1,
    reg_lambda        = 1.5,
    random_state      = SEED,
    n_jobs            = -1,
    verbosity         = -1,
)

oof_C  = np.zeros(len(train))
pred_C = np.zeros(len(test))
lgb_cbs = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]

for fold, (tr_i, val_i) in enumerate(skf.split(X_lgb, y_binned), 1):
    Xtr, Xval  = X_lgb.iloc[tr_i], X_lgb.iloc[val_i]
    ytr        = y_win.iloc[tr_i]
    y_raw_val  = y_raw.iloc[val_i]

    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xtr, ytr,
          eval_set=[(Xval, y_win.iloc[val_i])],
          callbacks=lgb_cbs,
          categorical_feature=CAT_FEATURES)

    oof_C[val_i] = m.predict(Xval)
    pred_C      += m.predict(X_test_lgb) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_C[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}")
    del m; gc.collect()

r2_C = r2_score(y_raw, oof_C)
print(f"\n  [Model C] OOF R² (original CLTV): {r2_C:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL D — XGBoost DART (NEW)
# DART randomly drops trees during boosting → reduces outlier dominance
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL D : XGBoost DART booster (winsorized CLTV)  [NEW]")
print("=" * 65)

# XGBoost needs purely numeric data — use ordinal encoding
oe_xgb       = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_xgb        = X.copy()
X_test_xgb   = X_test.copy()
X_xgb[CAT_FEATURES]      = oe_xgb.fit_transform(X_xgb[CAT_FEATURES].astype(str))
X_test_xgb[CAT_FEATURES] = oe_xgb.transform(X_test_xgb[CAT_FEATURES].astype(str))

XGB_PARAMS = dict(
    booster           = "dart",
    n_estimators      = 2000,
    learning_rate     = 0.05,
    max_depth         = 8,
    min_child_weight  = 20,             # high = more conservative splits
    subsample         = 0.75,
    colsample_bytree  = 0.75,
    colsample_bylevel = 0.75,
    gamma             = 1.0,            # min_split_loss = pruning regularizer
    reg_alpha         = 0.1,
    reg_lambda        = 2.0,
    rate_drop         = 0.1,            # DART: 10% of trees dropped each round
    skip_drop         = 0.5,            # DART: 50% chance to skip drop
    random_state      = SEED,
    n_jobs            = -1,
    tree_method       = "hist",
    verbosity         = 0,
)

oof_D  = np.zeros(len(train))
pred_D = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X_xgb, y_binned), 1):
    Xtr, Xval   = X_xgb.iloc[tr_i].values, X_xgb.iloc[val_i].values
    ytr         = y_win.iloc[tr_i].values
    y_raw_val   = y_raw.iloc[val_i]

    # DART does NOT support early stopping reliably — fix n_estimators
    m = xgb.XGBRegressor(**XGB_PARAMS)
    m.fit(Xtr, ytr, verbose=False)

    oof_D[val_i] = m.predict(Xval)
    pred_D      += m.predict(X_test_xgb.values) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_D[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}")
    del m; gc.collect()

r2_D = r2_score(y_raw, oof_D)
print(f"\n  [Model D] OOF R² (original CLTV): {r2_D:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL E — ExtraTrees on FULL (non-winsorized) CLTV (NEW)
# High variance reduction via extreme randomization.
# Trained on FULL target so the stacker gets a non-clipped signal.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL E : ExtraTrees on FULL CLTV (variance reduction)  [NEW]")
print("=" * 65)

X_et      = X_hgb.copy()   # already ordinal-encoded
X_test_et = X_test_hgb.copy()

ET_PARAMS = dict(
    n_estimators      = 500,
    max_features      = 0.5,
    min_samples_leaf  = 10,
    max_depth         = None,
    random_state      = SEED,
    n_jobs            = -1,
    bootstrap         = True,
)

oof_E  = np.zeros(len(train))
pred_E = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X_et, y_binned), 1):
    Xtr, Xval   = X_et.iloc[tr_i].values, X_et.iloc[val_i].values
    ytr         = y_raw.iloc[tr_i].values   # FULL y, not winsorized
    y_raw_val   = y_raw.iloc[val_i]

    m = ExtraTreesRegressor(**ET_PARAMS)
    m.fit(Xtr, ytr)

    oof_E[val_i] = m.predict(Xval)
    pred_E      += m.predict(X_test_et.values) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_E[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}")
    del m; gc.collect()

r2_E = r2_score(y_raw, oof_E)
print(f"\n  [Model E] OOF R² (original CLTV): {r2_E:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL F — LightGBM Quantile Regression p=0.5 (NEW)
# Predicts the conditional median — robust to outliers by construction.
# Its OOF predictions add a qualitatively different signal to the stacker.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL F : LightGBM Quantile (p=0.5)  [NEW]")
print("=" * 65)

LGB_Q_PARAMS = dict(
    objective         = "quantile",
    alpha             = 0.5,            # median regression
    metric            = "quantile",
    n_estimators      = 2000,
    learning_rate     = 0.04,
    num_leaves        = 127,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.7,
    reg_alpha         = 0.1,
    reg_lambda        = 1.0,
    random_state      = SEED,
    n_jobs            = -1,
    verbosity         = -1,
)

oof_F  = np.zeros(len(train))
pred_F = np.zeros(len(test))
lgb_cbs_q = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)]

for fold, (tr_i, val_i) in enumerate(skf.split(X_lgb, y_binned), 1):
    Xtr, Xval  = X_lgb.iloc[tr_i], X_lgb.iloc[val_i]
    ytr        = y_raw.iloc[tr_i]   # train on full CLTV for quantile
    y_raw_val  = y_raw.iloc[val_i]

    m = lgb.LGBMRegressor(**LGB_Q_PARAMS)
    m.fit(Xtr, ytr,
          eval_set=[(Xval, y_raw.iloc[val_i])],
          callbacks=lgb_cbs_q,
          categorical_feature=CAT_FEATURES)

    oof_F[val_i] = m.predict(Xval)
    pred_F      += m.predict(X_test_lgb) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_F[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}")
    del m; gc.collect()

r2_F = r2_score(y_raw, oof_F)
print(f"\n  [Model F] OOF R² (original CLTV): {r2_F:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STACKING — Ridge meta-learner (level-2)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STACKING : Ridge meta-learner (6 base models)")
print("=" * 65)

oof_stack  = np.column_stack([oof_A, oof_B, oof_C, oof_D, oof_E, oof_F])
test_stack = np.column_stack([pred_A, pred_B, pred_C, pred_D, pred_E, pred_F])
labels     = ["CatBoost", "HistGB", "LGB-RMSE", "XGB-DART", "ExtraTrees", "LGB-Q50"]

# Ridge meta-learner
ridge = Ridge(alpha=10.0, fit_intercept=True)
ridge.fit(oof_stack, y_raw)
pred_meta = ridge.predict(test_stack)
oof_meta  = ridge.predict(oof_stack)
r2_meta   = r2_score(y_raw, oof_meta)

print(f"\n  Ridge coefficients:")
for name, coef in zip(labels, ridge.coef_):
    print(f"    {name:12s} → {coef:.4f}")
print(f"  Intercept : {ridge.intercept_:.0f}")
print(f"  [Stack] OOF R² : {r2_meta:.5f}")

# Also try simple optimized blend as backup
print("\n  Running optimized blend search …")
best_r2, best_ws = -99, None
step = 0.1
candidates = np.arange(0.0, 1.01, step)
# Random search over weight space (faster than nested loops for 6 models)
np.random.seed(SEED)
for _ in range(5000):
    w = np.random.dirichlet(np.ones(6))
    r2 = r2_score(y_raw, oof_stack @ w)
    if r2 > best_r2:
        best_r2, best_ws = r2, w.copy()

r2_blend   = best_r2
pred_blend = test_stack @ best_ws
print(f"  Best blend R² : {r2_blend:.5f}  "
      f"(w: {dict(zip(labels, [f'{x:.2f}' for x in best_ws]))})")

# Final selection
if r2_meta >= r2_blend:
    pred_final    = pred_meta
    final_label   = f"Ridge stack  R²={r2_meta:.5f}"
    best_final_r2 = r2_meta
else:
    pred_final    = pred_blend
    final_label   = f"Blend        R²={r2_blend:.5f}"
    best_final_r2 = r2_blend

print(f"\n  → Chosen : {final_label}")

# ─── POST-PROCESSING ──────────────────────────────────────────────────────────
pred_final = np.clip(pred_final, cltv_floor, cltv_true_max)
print(f"\n  Final pred range : [{pred_final.min():.0f}, {pred_final.max():.0f}]")

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
print("\n[Step] Feature importance from CatBoost …")
fi_m = CatBoostRegressor(**{**CB_PARAMS, "iterations": 2000, "verbose": 0})
fi_m.fit(Pool(X, y_win, cat_features=cat_idx))

fi_df = (pd.DataFrame({"feature": ALL_FEATURES,
                        "importance": fi_m.get_feature_importance()})
           .sort_values("importance", ascending=False))
print("\n--- Top 30 Features ---")
print(fi_df.head(30).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 10))
sns.barplot(data=fi_df.head(30), x="importance", y="feature",
            palette="Blues_r", ax=ax)
ax.set_title("Top 30 Features — CatBoost v5 (Winsorized CLTV)")
plt.tight_layout()
plt.savefig("plots/feature_importance_v5.png", dpi=120)
plt.close()
print("  → plots/feature_importance_v5.png saved")

# ─── SUBMISSION ───────────────────────────────────────────────────────────────
submission = pd.DataFrame({
    "id"  : test["id"],
    "cltv": pred_final.astype(int)
})
submission.to_csv("submission_v5.csv", index=False)
print(f"\n[Saved] submission_v5.csv  ({len(submission)} rows)")
print(submission.head(10).to_string(index=False))

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
elapsed = (time.time() - START) / 60
print("\n" + "=" * 65)
print("  FINAL SUMMARY  (OOF R² on ORIGINAL CLTV)")
print("=" * 65)
print(f"  Model A — CatBoost  (depth=9, win)   : {r2_A:.5f}")
print(f"  Model B — HistGB    (depth=9, win)   : {r2_B:.5f}")
print(f"  Model C — LightGBM  (255L,  win)     : {r2_C:.5f}")
print(f"  Model D — XGBoost   (DART,  win)     : {r2_D:.5f}  [NEW]")
print(f"  Model E — ExtraTrees (full CLTV)     : {r2_E:.5f}  [NEW]")
print(f"  Model F — LGB-Q50   (quantile)       : {r2_F:.5f}  [NEW]")
print(f"  ────────────────────────────────────────────────────")
print(f"  Ridge stack (6 models)               : {r2_meta:.5f}")
print(f"  Optimized blend                      : {r2_blend:.5f}")
print(f"  ════════════════════════════════════════════════════")
print(f"  FINAL ({final_label})")
print(f"  Wall time : {elapsed:.1f} min")
print("=" * 65)
print("  submission_v5.csv ready  ✓")
print("=" * 65)