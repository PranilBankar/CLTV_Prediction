"""
==============================================================================
VahanBima CLTV Prediction — v2 (Improved Pipeline)
==============================================================================
ROOT CAUSE OF v1 LOW SCORE:
  • Training in log space but evaluating R² in original CLTV space
  • Jensen's inequality: E[exp(log_pred)] ≠ exp(E[log_pred])
  • Result: great log-space R² (0.32) but terrible orig-space R² (0.11)

FIXES IN v2:
  1. Train CatBoost / LGBM directly in ORIGINAL CLTV space → direct R² opt.
  2. Add CV-safe GROUP STATS (mean/median/std per segment) as features
  3. Log-space model + bias correction as 3rd model in blend
  4. Optimal weighted blending (grid-search weights)
==============================================================================
"""

# ─── 0. INSTALLS (uncomment in Colab) ────────────────────────────────────────
# import subprocess, sys
# for pkg in ["catboost", "lightgbm", "xgboost", "optuna"]:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

# ─── 1. IMPORTS ───────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import gc, os

SEED   = 42
N_FOLD = 5
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)

# ─── 2. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  LOADING DATA")
print("=" * 65)
train = pd.read_csv("Data/train_data.csv")
test  = pd.read_csv("Data/test_data.csv")
print(f"Train: {train.shape}  |  Test: {test.shape}")

TARGET = "cltv"

# ─── 3. INCOME ORDINAL MAP ────────────────────────────────────────────────────
# Preserves natural income ordering so model can capture monotone effects
INCOME_ORDER = {"<=2L": 0, "2L-5L": 1, "5L-10L": 2, "More than 10L": 3}

def build_income_map(s):
    return {v: INCOME_ORDER.get(v, i) for i, v in enumerate(sorted(s.unique()))}

income_map = build_income_map(pd.concat([train["income"], test["income"]]))
print("Income map:", income_map)

# ─── 4. BASE FEATURE ENGINEERING ─────────────────────────────────────────────
def feature_engineer(df, income_map):
    df = df.copy()

    # Ordinal income
    df["income_num"] = df["income"].map(income_map)

    # Multi-policy flag
    df["multi_policy_flag"] = (df["num_policies"] != "1").astype(int)

    # Claim-derived features
    df["claim_flag"]       = (df["claim_amount"] > 0).astype(int)
    df["log_claim"]        = np.log1p(df["claim_amount"])
    df["claim_per_yr"]     = df["claim_amount"] / (df["vintage"] + 1)
    df["log_claim_per_yr"] = np.log1p(df["claim_per_yr"])
    df["claim_sq"]         = df["claim_amount"] ** 0.5   # sqrt smoothing

    # vintage × claim
    df["vintage_x_log_claim"] = df["vintage"] * df["log_claim"]

    # Segmentation flags based on data quartiles
    df["high_income"] = (df["income_num"] >= 2).astype(int)
    df["high_claim"]  = (df["claim_amount"] > 6094).astype(int)   # 75th pct
    df["zero_claim"]  = (df["claim_amount"] == 0).astype(int)

    # Interaction strings (CatBoost handles these natively)
    df["policy_type"]         = df["policy"].astype(str) + "_" + df["type_of_policy"].astype(str)
    df["income_policy"]       = df["income"].astype(str) + "_" + df["policy"].astype(str)
    df["income_type"]         = df["income"].astype(str) + "_" + df["type_of_policy"].astype(str)
    df["area_qual"]           = df["area"].astype(str)   + "_" + df["qualification"].astype(str)
    df["gender_marital"]      = df["gender"].astype(str) + "_" + df["marital_status"].astype(str)
    df["income_area"]         = df["income"].astype(str) + "_" + df["area"].astype(str)
    df["policy_type_income"]  = df["policy_type"].astype(str) + "_" + df["income"].astype(str)
    df["income_vintage"]      = df["income"].astype(str) + "_" + df["vintage"].astype(str)
    df["claim_vintage"]       = df["high_claim"].astype(str) + "_" + df["vintage"].astype(str)

    # Vintage bucket
    df["vintage_bucket"] = pd.cut(
        df["vintage"], bins=[-1, 1, 3, 6, 100],
        labels=["new", "mid", "old", "very_old"]
    ).astype(str)

    # income × log_claim product
    df["income_x_claim"] = df["income_num"] * df["log_claim"]

    return df

train = feature_engineer(train, income_map)
test  = feature_engineer(test,  income_map)

# ─── 5. CV-SAFE TARGET ENCODING (on raw CLTV) ────────────────────────────────
# Encodes each categorical column as mean/median/std of CLTV per group,
# computed out-of-fold so there is ZERO target leakage.
# We encode raw CLTV (not log) because our evaluation metric is in orig space.

def cv_target_encode_stats(train_df, test_df, cat_cols, target_col,
                            n_splits=5, seed=42):
    """
    For each cat col, creates 3 new numeric columns:
      <col>_te_mean, <col>_te_median, <col>_te_std
    All computed out-of-fold for train; averaged over folds for test.
    """
    kf          = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    global_mean = train_df[target_col].mean()
    global_med  = train_df[target_col].median()
    global_std  = train_df[target_col].std()

    out_train = pd.DataFrame(index=train_df.index)
    out_test  = pd.DataFrame(index=test_df.index)

    for col in cat_cols:
        mean_tr = np.full(len(train_df), np.nan)
        med_tr  = np.full(len(train_df), np.nan)
        std_tr  = np.full(len(train_df), np.nan)

        mean_te_folds = np.zeros((len(test_df), n_splits))
        med_te_folds  = np.zeros((len(test_df), n_splits))
        std_te_folds  = np.zeros((len(test_df), n_splits))

        for fold, (tr_idx, val_idx) in enumerate(kf.split(train_df)):
            tr_fold  = train_df.iloc[tr_idx]
            val_fold = train_df.iloc[val_idx]

            grp = tr_fold.groupby(col)[target_col]
            map_mean = grp.mean();   map_med = grp.median();   map_std = grp.std()

            mean_tr[val_idx] = val_fold[col].map(map_mean).fillna(global_mean).values
            med_tr[val_idx]  = val_fold[col].map(map_med).fillna(global_med).values
            std_tr[val_idx]  = val_fold[col].map(map_std).fillna(global_std).values

            mean_te_folds[:, fold] = test_df[col].map(map_mean).fillna(global_mean).values
            med_te_folds[:, fold]  = test_df[col].map(map_med).fillna(global_med).values
            std_te_folds[:, fold]  = test_df[col].map(map_std).fillna(global_std).values

        out_train[col + "_te_mean"]   = np.where(np.isnan(mean_tr), global_mean, mean_tr)
        out_train[col + "_te_median"] = np.where(np.isnan(med_tr),  global_med,  med_tr)
        out_train[col + "_te_std"]    = np.where(np.isnan(std_tr),  global_std,  std_tr)

        out_test[col + "_te_mean"]   = mean_te_folds.mean(axis=1)
        out_test[col + "_te_median"] = med_te_folds.mean(axis=1)
        out_test[col + "_te_std"]    = std_te_folds.mean(axis=1)

    return out_train, out_test


print("\n[Step] CV-safe group statistics encoding (mean/median/std of CLTV)…")

# Encode key categorical combinations that have the most group-level signal
TE_COLS = [
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy",
    "policy_type", "income_policy", "income_type",
    "area_qual", "income_area", "policy_type_income",
    "vintage_bucket", "income_vintage",
]

te_train, te_test = cv_target_encode_stats(
    train, test, TE_COLS, target_col=TARGET, n_splits=N_FOLD, seed=SEED
)
train = pd.concat([train, te_train], axis=1)
test  = pd.concat([test,  te_test],  axis=1)

TE_STAT_COLS = te_train.columns.tolist()
print(f"  → Added {len(TE_STAT_COLS)} group-stat columns")

# ─── 6. DEFINE FEATURE SETS ───────────────────────────────────────────────────
CAT_FEATURES = [
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy", "num_policies",
    "policy_type", "income_policy", "income_type",
    "area_qual", "gender_marital", "income_area",
    "policy_type_income", "income_vintage", "claim_vintage",
    "vintage_bucket",
]

NUM_FEATURES = [
    "marital_status", "vintage", "claim_amount",
    "income_num", "multi_policy_flag",
    "claim_flag", "zero_claim", "log_claim",
    "claim_per_yr", "log_claim_per_yr", "claim_sq",
    "high_income", "high_claim",
    "income_x_claim", "vintage_x_log_claim",
] + TE_STAT_COLS

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
print(f"  Total features: {len(ALL_FEATURES)}")

# ─── 7. PREPARE DATA ──────────────────────────────────────────────────────────
X      = train[ALL_FEATURES].copy()
y      = train[TARGET].astype(float)        # ← ORIGINAL CLTV (no log!)
y_log  = np.log1p(y)                        # keep for log-space model
X_test = test[ALL_FEATURES].copy()

# CatBoost needs string categoricals
for col in CAT_FEATURES:
    X[col]      = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]

# Stratify on binned target for balanced folds
y_binned = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

# ─── 8. MODEL A — CatBoost in ORIGINAL SPACE ─────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL A: CatBoost (original CLTV space)")
print("=" * 65)

CB_PARAMS_ORIG = {
    "iterations"         : 5000,
    "learning_rate"      : 0.03,
    "depth"              : 7,
    "l2_leaf_reg"        : 3.0,
    "bagging_temperature": 0.5,
    "random_strength"    : 1.0,
    "border_count"       : 128,
    "loss_function"      : "RMSE",        # minimise RMSE of raw CLTV
    "eval_metric"        : "RMSE",
    "od_type"            : "Iter",
    "od_wait"            : 300,
    "random_seed"        : SEED,
    "verbose"            : 0,
    "thread_count"       : -1,
}

oof_A  = np.zeros(len(train))
pred_A = np.zeros(len(test))

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_binned), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    tr_pool  = Pool(X_tr, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    m = CatBoostRegressor(**CB_PARAMS_ORIG)
    m.fit(tr_pool, eval_set=val_pool, use_best_model=True)

    oof_A[val_idx] = m.predict(X_val)
    pred_A        += m.predict(X_test) / N_FOLD

    r2 = r2_score(y_val, oof_A[val_idx])
    print(f"  Fold {fold}/{N_FOLD}  R²: {r2:.5f}  (best iter: {m.best_iteration_})")
    del m, tr_pool, val_pool; gc.collect()

r2_A = r2_score(y, oof_A)
print(f"\n  [Model A] OOF R² (orig space): {r2_A:.5f}")

# ─── 9. MODEL B — LightGBM in ORIGINAL SPACE ─────────────────────────────────
print("\n" + "=" * 65)
print("  MODEL B: LightGBM (original CLTV space)")
print("=" * 65)

X_lgb      = X.copy();     X_test_lgb = X_test.copy()
for col in CAT_FEATURES:
    X_lgb[col]      = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

LGB_PARAMS = {
    "objective"        : "regression",
    "metric"           : "rmse",
    "n_estimators"     : 5000,
    "learning_rate"    : 0.03,
    "num_leaves"       : 255,
    "min_child_samples": 20,
    "subsample"        : 0.8,
    "colsample_bytree" : 0.8,
    "reg_alpha"        : 0.05,
    "reg_lambda"       : 1.0,
    "random_state"     : SEED,
    "n_jobs"           : -1,
    "verbosity"        : -1,
}

oof_B  = np.zeros(len(train))
pred_B = np.zeros(len(test))
cbs    = [lgb.early_stopping(300, verbose=False), lgb.log_evaluation(-1)]

for fold, (tr_idx, val_idx) in enumerate(skf.split(X_lgb, y_binned), 1):
    X_tr, X_val = X_lgb.iloc[tr_idx], X_lgb.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
          callbacks=cbs, categorical_feature=CAT_FEATURES)

    oof_B[val_idx] = m.predict(X_val)
    pred_B        += m.predict(X_test_lgb) / N_FOLD

    r2 = r2_score(y_val, oof_B[val_idx])
    print(f"  Fold {fold}/{N_FOLD}  R²: {r2:.5f}")
    del m; gc.collect()

r2_B = r2_score(y, oof_B)
print(f"\n  [Model B] OOF R² (orig space): {r2_B:.5f}")

# ─── 10. MODEL C — CatBoost in LOG SPACE + BIAS CORRECTION ───────────────────
# Jensen's inequality: E[exp(log_pred)] < exp(E[log_pred])
# Correction: pred_orig = exp(pred_log + 0.5 * residual_var)
# residual_var is estimated from OOF residuals in log space.

print("\n" + "=" * 65)
print("  MODEL C: CatBoost (log space + bias correction)")
print("=" * 65)

CB_PARAMS_LOG = {**CB_PARAMS_ORIG,
                 "learning_rate": 0.03,
                 "depth"        : 7}

oof_C_log  = np.zeros(len(train))
pred_C_log = np.zeros(len(test))
oof_resid_var = []  # per-fold residual variance in log space

for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y_binned), 1):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y_log.iloc[tr_idx], y_log.iloc[val_idx]

    tr_pool  = Pool(X_tr, y_tr, cat_features=cat_idx)
    val_pool = Pool(X_val, y_val, cat_features=cat_idx)

    m = CatBoostRegressor(**CB_PARAMS_LOG)
    m.fit(tr_pool, eval_set=val_pool, use_best_model=True)

    oof_C_log[val_idx] = m.predict(X_val)
    pred_C_log        += m.predict(X_test) / N_FOLD

    # Residual variance for bias correction
    resid = y_val.values - oof_C_log[val_idx]
    oof_resid_var.append(np.var(resid))

    r2_log = r2_score(y_val, oof_C_log[val_idx])
    r2_orig = r2_score(np.expm1(y_val), np.expm1(oof_C_log[val_idx]))
    print(f"  Fold {fold}/{N_FOLD}  R²(log): {r2_log:.5f}  |  R²(orig, raw): {r2_orig:.5f}")
    del m, tr_pool, val_pool; gc.collect()

# Apply log-normal bias correction:  E[y] = exp(μ_pred + σ²/2)
mean_resid_var = np.mean(oof_resid_var)
print(f"\n  Mean residual variance in log space: {mean_resid_var:.5f}")
print(f"  Bias correction factor: exp(+{mean_resid_var/2:.5f})")

oof_C  = np.expm1(oof_C_log  + 0.5 * mean_resid_var)
pred_C = np.expm1(pred_C_log + 0.5 * mean_resid_var)

r2_C = r2_score(y, oof_C)
print(f"\n  [Model C] OOF R² (orig space, bias-corrected): {r2_C:.5f}")

# ─── 11. OPTIMAL 3-WAY BLEND ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  OPTIMAL 3-WAY BLEND (A + B + C)")
print("=" * 65)

# Grid-search over (wA, wB, wC) such that wA+wB+wC=1, all ≥ 0
best_r2, best_w = -99, (0.4, 0.4, 0.2)
step = 0.05
for wA in np.arange(0, 1.0 + step, step):
    for wB in np.arange(0, 1.0 - wA + step, step):
        wC = round(1.0 - wA - wB, 6)
        if wC < 0: continue
        blend = wA * oof_A + wB * oof_B + wC * oof_C
        r2 = r2_score(y, blend)
        if r2 > best_r2:
            best_r2, best_w = r2, (wA, wB, wC)

wA, wB, wC = best_w
pred_blend = wA * pred_A + wB * pred_B + wC * pred_C
oof_blend  = wA * oof_A  + wB * oof_B  + wC * oof_C

print(f"  Weights  →  A(CB-orig): {wA:.2f}  |  B(LGB-orig): {wB:.2f}  |  C(CB-log+bias): {wC:.2f}")
print(f"  Blended OOF R²: {best_r2:.5f}")

# ─── 12. POST-PROCESSING — CLIP TO TRAINING RANGE ────────────────────────────
cltv_min = train[TARGET].min()
cltv_max = train[TARGET].max()
pred_final = np.clip(pred_blend, cltv_min, cltv_max)

print(f"\n  Pred range after clipping: [{pred_final.min():.0f}, {pred_final.max():.0f}]")

# ─── 13. FEATURE IMPORTANCE ───────────────────────────────────────────────────
print("\n[Step] Feature importance …")
fi_m = CatBoostRegressor(**{**CB_PARAMS_ORIG, "iterations": 2000, "verbose": 0})
fi_m.fit(Pool(X, y, cat_features=cat_idx))
fi_df = pd.DataFrame({
    "feature":    ALL_FEATURES,
    "importance": fi_m.get_feature_importance(),
}).sort_values("importance", ascending=False)

print("\n--- Top 25 Features ---")
print(fi_df.head(25).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 9))
sns.barplot(data=fi_df.head(25), x="importance", y="feature", palette="Blues_r", ax=ax)
ax.set_title("Top 25 Feature Importances (CatBoost, Original Space)")
plt.tight_layout()
plt.savefig("plots/feature_importance_v2.png", dpi=120)
plt.close()
print("[Saved] plots/feature_importance_v2.png")

# ─── 14. SUBMISSION ───────────────────────────────────────────────────────────
submission = pd.DataFrame({
    "id"  : test["id"],
    "cltv": pred_final.astype(int),
})
submission.to_csv("submission_v2.csv", index=False)
print(f"\n[Saved] submission_v2.csv  ({len(submission)} rows)")
print(submission.head(10).to_string(index=False))

# ─── 15. FINAL SUMMARY ────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  FINAL SUMMARY (all R² in ORIGINAL CLTV space)")
print("=" * 65)
print(f"  Model A — CatBoost (orig)          : {r2_A:.5f}")
print(f"  Model B — LightGBM (orig)          : {r2_B:.5f}")
print(f"  Model C — CatBoost (log+bias corr) : {r2_C:.5f}")
print(f"  Final Blend (A×{wA}+B×{wB}+C×{wC}) : {best_r2:.5f}")
print("=" * 65)
print("  submission_v2.csv is ready  ✓")
print("=" * 65)
