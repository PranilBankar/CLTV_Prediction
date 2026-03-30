"""
==============================================================================
VahanBima CLTV Prediction — v4 (Fast + Target Winsorization + Stacking)
==============================================================================
WHY v3 FAILED:
  • Huber(delta=50000) → all early residuals > delta → L1 from the start
    → glacially slow convergence, R² collapses to 0.09
  • 5000 iters × 4 models = hours of runtime

v4 STRATEGY:
  1. TARGET WINSORIZATION (99th pct) — the single biggest improvement lever.
     Top 1% extreme CLTV values are unlearnable with 11 features but dominate
     MSE loss. By capping during training, the model focuses on the 99%
     majority and learns much better boundaries.
  2. FAST CATBOOST (lr=0.05, iter=2000, od_wait=150) — same settings that
     gave R²=0.16 in v2 but faster.
  3. SKLEARN HistGradientBoosting — 5-10x faster than CatBoost/LGBM, adds
     diversity to the ensemble.
  4. RIDGE META-LEARNER (stacking) — learns optimal blend from OOF
     predictions, more principled than weight grid search.
  5. SAME PROVEN FEATURES from v2 — no extra noise.
==============================================================================
"""

# ─── INSTALLS (uncomment if needed) ──────────────────────────────────────────
# import subprocess, sys
# for pkg in ["catboost", "lightgbm"]:
#     subprocess.check_call([sys.executable, "-m", "pip", "install", pkg, "-q"])

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc

from sklearn.model_selection   import StratifiedKFold, KFold
from sklearn.metrics           import r2_score
from sklearn.ensemble          import HistGradientBoostingRegressor, ExtraTreesRegressor
from sklearn.linear_model      import Ridge
from sklearn.preprocessing     import OrdinalEncoder

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

SEED   = 42
N_FOLD = 5
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 60)
print("  LOADING DATA")
print("=" * 60)
train = pd.read_csv("Data/train_data.csv")
test  = pd.read_csv("Data/test_data.csv")
TARGET = "cltv"
print(f"Train: {train.shape}  |  Test: {test.shape}")
print(f"CLTV  → mean={train[TARGET].mean():.0f}  std={train[TARGET].std():.0f}  "
      f"max={train[TARGET].max():.0f}")

# ─── 2. TARGET WINSORIZATION ──────────────────────────────────────────────────
# Cap training target at 99th percentile.
# Goal: STOP the model wasting capacity on extreme unlearnable values.
# The top 1% capped values are post-processed back by clipping to train max.

WINSOR_PCT  = 99
cltv_cap    = np.percentile(train[TARGET], WINSOR_PCT)
cltv_floor  = train[TARGET].min()
cltv_true_max = train[TARGET].max()

print(f"\nWinsorization cap ({WINSOR_PCT}th pct): {cltv_cap:.0f}")
n_capped = (train[TARGET] > cltv_cap).sum()
print(f"Rows capped: {n_capped} ({100*n_capped/len(train):.1f}% of train)")

train["cltv_winsor"] = np.clip(train[TARGET], cltv_floor, cltv_cap)

# ─── 3. INCOME ORDINAL MAP ────────────────────────────────────────────────────
INCOME_ORDER = {"<=2L": 0, "2L-5L": 1, "5L-10L": 2, "More than 10L": 3}
all_inc = pd.concat([train["income"], test["income"]])
income_map = {v: INCOME_ORDER.get(v, i)
              for i, v in enumerate(sorted(all_inc.unique()))}
print(f"\nIncome map: {income_map}")

# ─── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
def feature_engineer(df, income_map):
    df = df.copy()

    df["income_num"]       = df["income"].map(income_map)
    df["multi_policy"]     = (df["num_policies"] != "1").astype(int)
    df["claim_flag"]       = (df["claim_amount"] > 0).astype(int)
    df["zero_claim"]       = (df["claim_amount"] == 0).astype(int)
    df["log_claim"]        = np.log1p(df["claim_amount"])
    df["sqrt_claim"]       = np.sqrt(df["claim_amount"])
    df["claim_per_yr"]     = df["claim_amount"] / (df["vintage"] + 1)
    df["log_claim_per_yr"] = np.log1p(df["claim_per_yr"])
    df["vintage_sq"]       = df["vintage"] ** 2
    df["v_x_logclaim"]     = df["vintage"] * df["log_claim"]
    df["income_x_claim"]   = df["income_num"] * df["log_claim"]
    df["income_x_vintage"] = df["income_num"] * df["vintage"]
    df["high_income"]      = (df["income_num"] >= 2).astype(int)
    df["high_claim"]       = (df["claim_amount"] > 6094).astype(int)

    # Interaction strings → CatBoost handles natively
    df["policy_type"]   = df["policy"].astype(str) + "_" + df["type_of_policy"].astype(str)
    df["inc_policy"]    = df["income"].astype(str)  + "_" + df["policy"].astype(str)
    df["inc_type"]      = df["income"].astype(str)  + "_" + df["type_of_policy"].astype(str)
    df["area_qual"]     = df["area"].astype(str)    + "_" + df["qualification"].astype(str)
    df["inc_area"]      = df["income"].astype(str)  + "_" + df["area"].astype(str)
    df["pol_type_inc"]  = df["policy_type"].astype(str) + "_" + df["income"].astype(str)
    df["inc_vint"]      = df["income"].astype(str)  + "_" + df["vintage"].astype(str)
    df["pol_vint"]      = df["policy_type"].astype(str) + "_" + df["vintage"].astype(str)
    df["gen_mar"]       = df["gender"].astype(str)  + "_" + df["marital_status"].astype(str)

    df["vintage_bucket"] = pd.cut(df["vintage"], bins=[-1,1,3,6,100],
                                  labels=["new","mid","old","very_old"]).astype(str)
    df["claim_bin"]      = pd.qcut(df["claim_amount"], q=10,
                                   labels=False, duplicates="drop").astype(str)
    return df

train = feature_engineer(train, income_map)
test  = feature_engineer(test,  income_map)

# ─── 5. CLAIM PERCENTILE RANK (within segment) ───────────────────────────────
# Tells model: "how big is this claim RELATIVE to peers with same income/policy?"
n_tr = len(train)
tr_r = train.reset_index(drop=True)
te_r = test.reset_index(drop=True)
all_ = pd.concat([tr_r, te_r], axis=0, ignore_index=True)

RANK_SEGS = {
    "rank_in_income"  : ["income"],
    "rank_in_policy"  : ["policy_type"],
    "rank_in_inc_pol" : ["income", "policy_type"],
}
for col, grp in RANK_SEGS.items():
    all_[col] = all_.groupby(grp)["claim_amount"].rank(pct=True)
    train[col] = all_.iloc[:n_tr][col].values
    test[col]  = all_.iloc[n_tr:][col].values

RANK_COLS = list(RANK_SEGS.keys())

# ─── 6. CV-SAFE TARGET ENCODING (mean + std of WINSORIZED CLTV) ──────────────
print("\n[Step] CV-safe target encoding …")

TE_SEGS = [
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy",
    "policy_type", "inc_policy", "inc_type",
    "area_qual", "inc_area", "pol_type_inc", "inc_vint",
]

def cv_te(train_df, test_df, cols, target, n_splits=5, seed=42):
    kf      = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gmean   = train_df[target].mean()
    gstd    = train_df[target].std()
    out_tr  = pd.DataFrame(index=train_df.index)
    out_te  = pd.DataFrame(index=test_df.index)

    for col in cols:
        mean_arr  = np.full(len(train_df), np.nan)
        std_arr   = np.full(len(train_df), np.nan)
        mean_te_f = np.zeros((len(test_df), n_splits))
        std_te_f  = np.zeros((len(test_df), n_splits))

        for f, (tri, vali) in enumerate(kf.split(train_df)):
            tr_f  = train_df.iloc[tri]
            val_f = train_df.iloc[vali]
            grp   = tr_f.groupby(col)[target]
            mm = grp.mean(); ms = grp.std()

            mean_arr[vali]   = val_f[col].map(mm).fillna(gmean).values
            std_arr[vali]    = val_f[col].map(ms).fillna(gstd).values
            mean_te_f[:, f]  = test_df[col].map(mm).fillna(gmean).values
            std_te_f[:, f]   = test_df[col].map(ms).fillna(gstd).values

        out_tr[col + "_te"]     = np.where(np.isnan(mean_arr), gmean, mean_arr)
        out_tr[col + "_te_std"] = np.where(np.isnan(std_arr),  gstd,  std_arr)
        out_te[col + "_te"]     = mean_te_f.mean(axis=1)
        out_te[col + "_te_std"] = std_te_f.mean(axis=1)

    return out_tr, out_te

# Encode on WINSORIZED CLTV — avoids extreme groups from distorting encoding
te_train, te_test = cv_te(train, test, TE_SEGS, "cltv_winsor", N_FOLD, SEED)
train = pd.concat([train, te_train], axis=1)
test  = pd.concat([test,  te_test],  axis=1)
TE_COLS = te_train.columns.tolist()
print(f"  → {len(TE_COLS)} group-stat columns")

# ─── 7. DEFINE FEATURES ───────────────────────────────────────────────────────
CAT_FEATURES = [
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy", "num_policies",
    "policy_type", "inc_policy", "inc_type",
    "area_qual", "inc_area", "pol_type_inc",
    "inc_vint", "pol_vint", "gen_mar",
    "vintage_bucket", "claim_bin",
]

NUM_FEATURES = [
    "marital_status", "vintage", "vintage_sq", "claim_amount",
    "income_num", "multi_policy",
    "claim_flag", "zero_claim", "log_claim", "sqrt_claim",
    "claim_per_yr", "log_claim_per_yr",
    "high_income", "high_claim",
    "income_x_claim", "income_x_vintage", "v_x_logclaim",
] + RANK_COLS + TE_COLS

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
print(f"\n  CAT: {len(CAT_FEATURES)}  NUM: {len(NUM_FEATURES)}  TOTAL: {len(ALL_FEATURES)}")

# ─── 8. PREPARE ARRAYS ────────────────────────────────────────────────────────
X      = train[ALL_FEATURES].copy()
y_raw  = train[TARGET].astype(float)          # original CLTV (for scoring)
y_win  = train["cltv_winsor"].astype(float)   # winsorized CLTV (for training)
X_test = test[ALL_FEATURES].copy()

for col in CAT_FEATURES:
    X[col]      = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]

# Stratify on log-binned ORIGINAL cltv so folds are balanced
y_binned = pd.qcut(np.log1p(y_raw), q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

# ─── 9. MODEL A — CatBoost RMSE on Winsorized CLTV ───────────────────────────
print("\n" + "=" * 60)
print("  MODEL A: CatBoost RMSE on winsorized CLTV  (FAST)")
print("=" * 60)

CB_PARAMS = {
    "iterations"         : 2000,
    "learning_rate"      : 0.05,     # higher LR = faster convergence
    "depth"              : 7,
    "l2_leaf_reg"        : 3.0,
    "bagging_temperature": 0.4,
    "random_strength"    : 0.8,
    "border_count"       : 128,
    "loss_function"      : "RMSE",
    "eval_metric"        : "RMSE",
    "od_type"            : "Iter",
    "od_wait"            : 150,      # faster early stopping
    "random_seed"        : SEED,
    "verbose"            : 0,
    "thread_count"       : -1,
}

oof_A  = np.zeros(len(train))
pred_A = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X, y_binned), 1):
    Xtr, Xval = X.iloc[tr_i], X.iloc[val_i]
    ytr, yval = y_win.iloc[tr_i], y_win.iloc[val_i]        # train on winsorized
    y_raw_val = y_raw.iloc[val_i]                           # score on original

    tr_pool  = Pool(Xtr,  ytr,  cat_features=cat_idx)
    val_pool = Pool(Xval, yval, cat_features=cat_idx)

    m = CatBoostRegressor(**CB_PARAMS)
    m.fit(tr_pool, eval_set=val_pool, use_best_model=True)

    oof_A[val_i] = m.predict(Xval)
    pred_A      += m.predict(X_test) / N_FOLD

    r2_win  = r2_score(yval,      oof_A[val_i])
    r2_orig = r2_score(y_raw_val, oof_A[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(winsor)={r2_win:.4f}  "
          f"R²(orig)={r2_orig:.4f}  iter={m.best_iteration_}")
    del m, tr_pool, val_pool; gc.collect()

r2_A = r2_score(y_raw, oof_A)
print(f"\n  [Model A] OOF R² on ORIGINAL CLTV: {r2_A:.5f}")

# ─── 10. MODEL B — HistGradientBoosting (fast sklearn model) ─────────────────
# 5-10x faster than CatBoost/LGBM, handles missing values natively,
# uses bin-based histogram splitting (same idea as LGBM but from sklearn).
print("\n" + "=" * 60)
print("  MODEL B: HistGradientBoosting (sklearn — fast)")
print("=" * 60)

# For HistGB we need ordinal-encoded categoricals (it supports cat natively)
oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_hgb      = X.copy()
X_test_hgb = X_test.copy()
X_hgb[CAT_FEATURES]      = oe.fit_transform(X_hgb[CAT_FEATURES].astype(str))
X_test_hgb[CAT_FEATURES]  = oe.transform(X_test_hgb[CAT_FEATURES].astype(str))

HGB_PARAMS = dict(
    max_iter            = 1000,
    learning_rate       = 0.05,
    max_depth           = 7,
    min_samples_leaf    = 20,
    l2_regularization   = 1.0,
    max_bins            = 255,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 50,
    random_state        = SEED,
    categorical_features= list(range(len(CAT_FEATURES))),  # first N cols are cats
)

oof_B  = np.zeros(len(train))
pred_B = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X_hgb, y_binned), 1):
    Xtr, Xval    = X_hgb.iloc[tr_i], X_hgb.iloc[val_i]
    ytr          = y_win.iloc[tr_i]                  # train on winsorized
    y_raw_val    = y_raw.iloc[val_i]

    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    m.fit(Xtr, ytr)

    oof_B[val_i] = m.predict(Xval)
    pred_B      += m.predict(X_test_hgb) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_B[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}  iter={m.n_iter_}")
    del m; gc.collect()

r2_B = r2_score(y_raw, oof_B)
print(f"\n  [Model B] OOF R² on ORIGINAL CLTV: {r2_B:.5f}")

# ─── 11. MODEL C — LightGBM RMSE on Winsorized CLTV ─────────────────────────
print("\n" + "=" * 60)
print("  MODEL C: LightGBM RMSE on winsorized CLTV")
print("=" * 60)

X_lgb      = X.copy()
X_test_lgb = X_test.copy()
for col in CAT_FEATURES:
    X_lgb[col]      = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

LGB_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 2000,
    learning_rate     = 0.05,
    num_leaves        = 127,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.05,
    reg_lambda        = 1.0,
    random_state      = SEED,
    n_jobs            = -1,
    verbosity         = -1,
)

oof_C  = np.zeros(len(train))
pred_C = np.zeros(len(test))
lgb_cbs = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)]

for fold, (tr_i, val_i) in enumerate(skf.split(X_lgb, y_binned), 1):
    Xtr, Xval = X_lgb.iloc[tr_i], X_lgb.iloc[val_i]
    ytr       = y_win.iloc[tr_i]                     # train on winsorized
    y_raw_val = y_raw.iloc[val_i]

    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xtr, ytr,
          eval_set=[(Xval, y_win.iloc[val_i])],      # early stop on winsorized val
          callbacks=lgb_cbs,
          categorical_feature=CAT_FEATURES)

    oof_C[val_i] = m.predict(Xval)
    pred_C      += m.predict(X_test_lgb) / N_FOLD

    r2_orig = r2_score(y_raw_val, oof_C[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²(orig)={r2_orig:.4f}")
    del m; gc.collect()

r2_C = r2_score(y_raw, oof_C)
print(f"\n  [Model C] OOF R² on ORIGINAL CLTV: {r2_C:.5f}")

# ─── 12. STACKING (Ridge meta-learner) ───────────────────────────────────────
# Uses OOF predictions from A, B, C as features for a Ridge regression.
# More principled than weight grid-search: Ridge learns optimal combination
# while avoiding overfitting (L2 penalty).

print("\n" + "=" * 60)
print("  STACKING: Ridge meta-learner")
print("=" * 60)

# Level-1 OOF features: [pred_A, pred_B, pred_C]
oof_stack  = np.column_stack([oof_A, oof_B, oof_C])
test_stack = np.column_stack([pred_A, pred_B, pred_C])

# Train Ridge meta-learner on OOF predictions → original CLTV
ridge = Ridge(alpha=10.0, fit_intercept=True)
ridge.fit(oof_stack, y_raw)
pred_meta = ridge.predict(test_stack)

oof_meta  = ridge.predict(oof_stack)
r2_meta   = r2_score(y_raw, oof_meta)

print(f"  Ridge coefficients: A={ridge.coef_[0]:.3f}  "
      f"B={ridge.coef_[1]:.3f}  C={ridge.coef_[2]:.3f}")
print(f"  Intercept: {ridge.intercept_:.0f}")
print(f"  [Stack] OOF R² on ORIGINAL CLTV: {r2_meta:.5f}")

# Also try simple grid-search blend as backup
best_r2, best_ws = -99, (1/3, 1/3, 1/3)
step = 0.05
for wA in np.arange(0.0, 1.01, step):
    for wB in np.arange(0.0, 1.01 - wA, step):
        wC = round(1.0 - wA - wB, 6)
        if wC < 0: continue
        r2 = r2_score(y_raw, wA*oof_A + wB*oof_B + wC*oof_C)
        if r2 > best_r2:
            best_r2, best_ws = r2, (wA, wB, wC)

wA, wB, wC = best_ws
pred_blend = wA*pred_A + wB*pred_B + wC*pred_C
r2_blend   = r2_score(y_raw, wA*oof_A + wB*oof_B + wC*oof_C)
print(f"\n  Grid-blend R²: {r2_blend:.5f}  (w: A={wA:.2f} B={wB:.2f} C={wC:.2f})")

# Choose the better of Ridge stack vs grid blend
if r2_meta >= r2_blend:
    pred_final  = pred_meta
    final_label = f"Ridge stack (R²={r2_meta:.5f})"
    best_final_r2 = r2_meta
else:
    pred_final  = pred_blend
    final_label = f"Grid blend  (R²={r2_blend:.5f})"
    best_final_r2 = r2_blend

print(f"  → Choosing: {final_label}")

# ─── 13. POST-PROCESSING ──────────────────────────────────────────────────────
# Clip to observed training range (after winsor correction, preds can be low)
pred_final = np.clip(pred_final, cltv_floor, cltv_true_max)
print(f"\n  Final pred range: [{pred_final.min():.0f}, {pred_final.max():.0f}]")

# ─── 14. FEATURE IMPORTANCE ───────────────────────────────────────────────────
print("\n[Step] Feature importance (CatBoost on winsorized CLTV) …")
fi_m = CatBoostRegressor(**{**CB_PARAMS, "iterations": 1500, "verbose": 0})
fi_m.fit(Pool(X, y_win, cat_features=cat_idx))

fi_df = pd.DataFrame({"feature": ALL_FEATURES,
                       "importance": fi_m.get_feature_importance()}
                     ).sort_values("importance", ascending=False)
print("\n--- Top 25 Features ---")
print(fi_df.head(25).to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 9))
sns.barplot(data=fi_df.head(25), x="importance", y="feature",
            palette="Blues_r", ax=ax)
ax.set_title("Top 25 Features (CatBoost, Winsorized CLTV)")
plt.tight_layout()
plt.savefig("plots/feature_importance_v4.png", dpi=120)
plt.close()

# ─── 15. SUBMISSION ───────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "cltv": pred_final.astype(int)})
submission.to_csv("submission_v4.csv", index=False)
print(f"\n[Saved] submission_v4.csv  ({len(submission)} rows)")
print(submission.head(10).to_string(index=False))

# ─── 16. SUMMARY ──────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("  FINAL SUMMARY  (R² on ORIGINAL CLTV)")
print("=" * 60)
print(f"  Model A — CatBoost  (winsorized)  : {r2_A:.5f}")
print(f"  Model B — HistGB    (winsorized)  : {r2_B:.5f}")
print(f"  Model C — LightGBM  (winsorized)  : {r2_C:.5f}")
print(f"  Grid blend                        : {r2_blend:.5f}")
print(f"  Ridge stack                       : {r2_meta:.5f}")
print(f"  ─────────────────────────────────────────────")
print(f"  FINAL   ({final_label})")
print("=" * 60)
print("  submission_v4.csv is ready  ✓")
print("=" * 60)
