"""
==============================================================================
VahanBima CLTV Prediction — v6  (Clean & Focused)
==============================================================================

WHY v5 PLATEAUED AT 0.1613:
  • 95 features — most were noise that CatBoost already handles natively
  • XGBoost DART: 1 hr, R²=0.09  →  dropped
  • All 5+ models were 0.994 correlated → stacking gained nothing
  • depth=9 CatBoost was stopping at iter ~215-357/3000: overfit signal

ROOT CAUSE DIAGNOSIS:
  With 11 raw features, the true max R² (20 claim buckets × all cats) is
  ~0.367. We're at 0.16. The gap is not hyperparameters or complexity —
  it's lack of MODEL DIVERSITY. When models disagree on the right answer,
  stacking resolves that. When models agree (ρ=0.994), stacking does nothing.

WHAT'S NEW IN v6:
─────────────────────────────────────────────────────────────────────────────
1. FEATURE REDUCTION: ~21 features (down from ~95)
   Removed all polynomial, z-score, group-aggregate features that had no
   measurable CV impact. Fewer features = less noise for CatBoost's ordered
   target statistics to process.

2. CATBOOST LOSSGUIDE — Model A2  [most important change]
   Default CatBoost uses SymmetricTree: same split threshold at every node
   of a given depth level. grow_policy='Lossguide' builds LEAF-WISE trees
   (same style as LightGBM/XGBoost): each new leaf is whichever existing
   leaf reduces loss most. This creates fundamentally different tree shapes
   and error patterns. Expected correlation with A1: ~0.85–0.90 vs 0.994
   between different GBM families.

3. CATBOOST MAE — Model A3
   MAE loss minimises the conditional median rather than mean. Even after
   99th-pct winsorization, the upper 5% still inflates RMSE gradients.
   MAE assigns equal weight to all residuals → different leaf assignments
   → different OOF predictions → useful stacking signal.

4. CORRECT CATBOOST DEPTH = 6
   depth=9 models in v5 converged at ~250 iters. This means the extra
   depth was purely memorising training noise. depth=6 with od_wait=300
   generalises better and runs ~2× faster.

5. LIGHTGBM path_smooth=1.0
   Smooths leaf predictions toward their parent node value, reducing
   variance of individual leaf estimates on a high-variance target.

MODELS (5 total, fast):
  A1 — CatBoost SymmetricTree depth=6  RMSE  (winsorized)
  A2 — CatBoost Lossguide     leaves=64 RMSE (winsorized)  [key new]
  A3 — CatBoost SymmetricTree depth=6  MAE   (winsorized)  [new loss]
  B  — LightGBM               leaves=63 RMSE (winsorized)
  C  — HistGradientBoosting   depth=6        (winsorized)
  META — Ridge stacking

EXPECTED RUNTIME: ~35–55 min  (3 CatBoost + LGB + HistGB, 5 folds each)
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
import os, gc, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import StratifiedKFold, KFold
from sklearn.metrics          import r2_score
from sklearn.ensemble         import HistGradientBoostingRegressor
from sklearn.linear_model     import Ridge
from sklearn.preprocessing    import OrdinalEncoder

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

SEED   = 42
N_FOLD = 5
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)
START  = time.time()

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  CLTV v6  (Clean & Focused)")
print("=" * 65)

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
WINSOR_PCT    = 99
cltv_cap      = np.percentile(train[TARGET], WINSOR_PCT)
cltv_floor    = train[TARGET].min()
cltv_true_max = train[TARGET].max()
train["cltv_winsor"] = np.clip(train[TARGET], cltv_floor, cltv_cap)
n_capped = (train[TARGET] > cltv_cap).sum()
print(f"\nWinsorization cap ({WINSOR_PCT}th pct): {cltv_cap:.0f}  "
      f"({n_capped} rows = {100*n_capped/len(train):.1f}% capped)")

# ─── 3. FEATURE ENGINEERING ───────────────────────────────────────────────────
INCOME_MAP = {"<=2L": 1, "2L-5L": 2, "5L-10L": 3, "More than 10L": 4}

def make_features(df):
    d = df.copy()
    d["income_num"]       = d["income"].map(INCOME_MAP).astype(float)
    # is_multi is the single strongest predictor (corr=0.36 with CLTV)
    d["is_multi"]         = (d["num_policies"] != "1").astype(float)
    d["log_claim"]        = np.log1p(d["claim_amount"])
    d["claim_per_yr"]     = d["claim_amount"] / (d["vintage"] + 1)
    # is_multi × other features (corr 0.27–0.35)
    d["multi_x_claim"]    = d["is_multi"] * d["claim_amount"]
    d["multi_x_logclaim"] = d["is_multi"] * d["log_claim"]
    d["multi_x_vintage"]  = d["is_multi"] * d["vintage"]
    d["multi_x_income"]   = d["is_multi"] * d["income_num"]
    return d

train = make_features(train)
test  = make_features(test)

# ─── 4. CV-SAFE BAYESIAN TARGET ENCODING ──────────────────────────────────────
# Only 3 TE columns, each anchored on num_policies (corr=0.36–0.38).
# Bayesian smoothing (smoothing=30): rare groups pulled toward global mean.
# OOF computation ensures zero leakage.
print("\n[Step] Bayesian target encoding (3 groups, OOF, smoothing=30) …")

TE_GROUPS = [
    ["num_policies"],
    ["num_policies", "area", "income"],
    ["num_policies", "area", "income", "policy", "type_of_policy"],
]
SMOOTHING = 30
gmean     = train["cltv_winsor"].mean()
kf_te     = KFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

def bayesian_te_oof(train_df, test_df, grp_cols, target_col, smoothing=30):
    arr_tr   = np.full(len(train_df), gmean)
    folds_te = np.zeros((len(test_df), N_FOLD))

    for f, (tri, vali) in enumerate(kf_te.split(train_df)):
        tr_f   = train_df.iloc[tri]
        counts = tr_f.groupby(grp_cols)[target_col].count()
        means  = tr_f.groupby(grp_cols)[target_col].mean()
        smooth = (counts * means + smoothing * gmean) / (counts + smoothing)

        def lookup(row):
            key = tuple(row) if len(grp_cols) > 1 else row.iloc[0]
            return smooth.get(key, gmean)

        arr_tr[vali]   = train_df.iloc[vali][grp_cols].apply(lookup, axis=1).values
        folds_te[:, f] = test_df[grp_cols].apply(lookup, axis=1).values

    return arr_tr, folds_te.mean(axis=1)

te_names = []
for grp in TE_GROUPS:
    col = "te_" + "_".join(grp)
    tr_vals, te_vals = bayesian_te_oof(train, test, grp, "cltv_winsor")
    train[col] = tr_vals
    test[col]  = te_vals
    te_names.append(col)
    corr = np.corrcoef(tr_vals, train[TARGET])[0, 1]
    print(f"  {col:55s}  corr={corr:.4f}")

# ─── 5. DEFINE FEATURES ───────────────────────────────────────────────────────
# CatBoost handles these natively via ordered target statistics
CAT_FEATURES = [
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy", "num_policies",
]

# Numeric: only features with verified signal (corr > 0.10 or key interaction)
NUM_FEATURES = [
    "marital_status", "vintage", "claim_amount",
    "income_num", "is_multi",
    "log_claim", "claim_per_yr",
    "multi_x_claim", "multi_x_logclaim",
    "multi_x_vintage", "multi_x_income",
] + te_names

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]

print(f"\n  CAT: {len(CAT_FEATURES)}  "
      f"NUM: {len(NUM_FEATURES)}  "
      f"TOTAL: {len(ALL_FEATURES)}  "
      f"(v5 had ~95)")

# ─── 6. PREPARE ARRAYS ────────────────────────────────────────────────────────
X      = train[ALL_FEATURES].copy()
y_raw  = train[TARGET].astype(float)
y_win  = train["cltv_winsor"].astype(float)
X_test = test[ALL_FEATURES].copy()

for col in CAT_FEATURES:
    X[col]      = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

y_binned = pd.qcut(np.log1p(y_raw), q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A1 — CatBoost SymmetricTree  depth=6  RMSE
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL A1 : CatBoost SymmetricTree  depth=6  RMSE")
print("=" * 65)

CB_SYM = {
    "iterations"          : 3000,
    "learning_rate"       : 0.03,
    "depth"               : 6,
    "l2_leaf_reg"         : 3.0,
    "bagging_temperature" : 0.5,
    "random_strength"     : 1.0,
    "border_count"        : 254,
    "grow_policy"         : "SymmetricTree",
    "loss_function"       : "RMSE",
    "eval_metric"         : "RMSE",
    "od_type"             : "Iter",
    "od_wait"             : 300,
    "random_seed"         : SEED,
    "verbose"             : 0,
    "thread_count"        : -1,
}

oof_A1  = np.zeros(len(train))
pred_A1 = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X, y_binned), 1):
    Xtr, Xval = X.iloc[tr_i], X.iloc[val_i]
    ytr, yval = y_win.iloc[tr_i], y_win.iloc[val_i]
    tr_p  = Pool(Xtr,  ytr,  cat_features=cat_idx)
    val_p = Pool(Xval, yval, cat_features=cat_idx)
    m = CatBoostRegressor(**CB_SYM)
    m.fit(tr_p, eval_set=val_p, use_best_model=True)
    oof_A1[val_i] = m.predict(Xval)
    pred_A1      += m.predict(X_test) / N_FOLD
    r2 = r2_score(y_raw.iloc[val_i], oof_A1[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²={r2:.4f}  best_iter={m.best_iteration_}")
    del m, tr_p, val_p; gc.collect()

r2_A1 = r2_score(y_raw, oof_A1)
print(f"\n  [A1] OOF R²: {r2_A1:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A2 — CatBoost Lossguide  max_leaves=64  RMSE
#
# KEY CHANGE: Lossguide builds leaf-wise trees (not symmetric).
# Each iteration grows the leaf that reduces loss the most.
# This finds deep, narrow patterns that SymmetricTree misses.
# Correlation with A1 expected ~0.85–0.90 → genuine stacking benefit.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL A2 : CatBoost Lossguide  max_leaves=64  RMSE  [KEY]")
print("=" * 65)

CB_LG = {**CB_SYM,
         "grow_policy" : "Lossguide",
         "max_leaves"  : 64,            # replaces depth for Lossguide
         "random_seed" : SEED,
         }
# Remove depth key — not valid for Lossguide
CB_LG.pop("depth", None)

oof_A2  = np.zeros(len(train))
pred_A2 = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X, y_binned), 1):
    Xtr, Xval = X.iloc[tr_i], X.iloc[val_i]
    ytr, yval = y_win.iloc[tr_i], y_win.iloc[val_i]
    tr_p  = Pool(Xtr,  ytr,  cat_features=cat_idx)
    val_p = Pool(Xval, yval, cat_features=cat_idx)
    m = CatBoostRegressor(**CB_LG)
    m.fit(tr_p, eval_set=val_p, use_best_model=True)
    oof_A2[val_i] = m.predict(Xval)
    pred_A2      += m.predict(X_test) / N_FOLD
    r2 = r2_score(y_raw.iloc[val_i], oof_A2[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²={r2:.4f}  best_iter={m.best_iteration_}")
    del m, tr_p, val_p; gc.collect()

r2_A2 = r2_score(y_raw, oof_A2)
print(f"\n  [A2] OOF R²: {r2_A2:.5f}")
rho_12 = np.corrcoef(oof_A1, oof_A2)[0, 1]
print(f"  Correlation A1↔A2: {rho_12:.4f}  (target <0.92 for useful diversity)")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL A3 — CatBoost SymmetricTree  depth=6  MAE
#
# MAE minimises conditional median → less influenced by the upper tail
# even after winsorization. Different loss = different gradient updates
# = different leaf assignments = genuinely different OOF predictions.
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL A3 : CatBoost SymmetricTree  depth=6  MAE  [NEW]")
print("=" * 65)

CB_MAE = {**CB_SYM,
          "loss_function" : "MAE",
          "eval_metric"   : "MAE",
          "random_seed"   : SEED + 1,
          }

oof_A3  = np.zeros(len(train))
pred_A3 = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X, y_binned), 1):
    Xtr, Xval = X.iloc[tr_i], X.iloc[val_i]
    ytr, yval = y_win.iloc[tr_i], y_win.iloc[val_i]
    tr_p  = Pool(Xtr,  ytr,  cat_features=cat_idx)
    val_p = Pool(Xval, yval, cat_features=cat_idx)
    m = CatBoostRegressor(**CB_MAE)
    m.fit(tr_p, eval_set=val_p, use_best_model=True)
    oof_A3[val_i] = m.predict(Xval)
    pred_A3      += m.predict(X_test) / N_FOLD
    r2 = r2_score(y_raw.iloc[val_i], oof_A3[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²={r2:.4f}  best_iter={m.best_iteration_}")
    del m, tr_p, val_p; gc.collect()

r2_A3 = r2_score(y_raw, oof_A3)
print(f"\n  [A3] OOF R²: {r2_A3:.5f}")
print(f"  Correlation A1↔A3: {np.corrcoef(oof_A1, oof_A3)[0,1]:.4f}")
print(f"  Correlation A2↔A3: {np.corrcoef(oof_A2, oof_A3)[0,1]:.4f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B — LightGBM RMSE  path_smooth=1.0
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL B : LightGBM RMSE  (path_smooth=1.0)")
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
    num_leaves        = 63,
    min_child_samples = 20,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 0.05,
    reg_lambda        = 1.5,
    path_smooth       = 1.0,
    random_state      = SEED,
    n_jobs            = -1,
    verbosity         = -1,
)

oof_B  = np.zeros(len(train))
pred_B = np.zeros(len(test))
lgb_cbs = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]

for fold, (tr_i, val_i) in enumerate(skf.split(X_lgb, y_binned), 1):
    Xtr, Xval = X_lgb.iloc[tr_i], X_lgb.iloc[val_i]
    ytr       = y_win.iloc[tr_i]
    m = lgb.LGBMRegressor(**LGB_PARAMS)
    m.fit(Xtr, ytr,
          eval_set=[(Xval, y_win.iloc[val_i])],
          callbacks=lgb_cbs,
          categorical_feature=CAT_FEATURES)
    oof_B[val_i] = m.predict(Xval)
    pred_B      += m.predict(X_test_lgb) / N_FOLD
    r2 = r2_score(y_raw.iloc[val_i], oof_B[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²={r2:.4f}")
    del m; gc.collect()

r2_B = r2_score(y_raw, oof_B)
print(f"\n  [B] OOF R²: {r2_B:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL C — HistGradientBoosting  depth=6
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  MODEL C : HistGradientBoosting  depth=6  (sklearn)")
print("=" * 65)

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_hgb      = X.copy()
X_test_hgb = X_test.copy()
X_hgb[CAT_FEATURES]      = oe.fit_transform(X_hgb[CAT_FEATURES].astype(str))
X_test_hgb[CAT_FEATURES] = oe.transform(X_test_hgb[CAT_FEATURES].astype(str))

HGB_PARAMS = dict(
    max_iter            = 2000,
    learning_rate       = 0.04,
    max_depth           = 6,
    min_samples_leaf    = 20,
    l2_regularization   = 1.0,
    max_bins            = 255,
    early_stopping      = True,
    validation_fraction = 0.1,
    n_iter_no_change    = 60,
    random_state        = SEED,
    categorical_features= list(range(len(CAT_FEATURES))),
)

oof_C  = np.zeros(len(train))
pred_C = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X_hgb, y_binned), 1):
    Xtr, Xval = X_hgb.iloc[tr_i], X_hgb.iloc[val_i]
    ytr       = y_win.iloc[tr_i]
    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    m.fit(Xtr, ytr)
    oof_C[val_i] = m.predict(Xval)
    pred_C      += m.predict(X_test_hgb) / N_FOLD
    r2 = r2_score(y_raw.iloc[val_i], oof_C[val_i])
    print(f"  Fold {fold}/{N_FOLD}  R²={r2:.4f}  n_iter={m.n_iter_}")
    del m; gc.collect()

r2_C = r2_score(y_raw, oof_C)
print(f"\n  [C] OOF R²: {r2_C:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# STACKING
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("  STACKING : Ridge meta-learner (5 models)")
print("=" * 65)

labels     = ["CB-Sym-RMSE", "CB-LG-RMSE", "CB-Sym-MAE", "LGB", "HistGB"]
oof_stack  = np.column_stack([oof_A1, oof_A2, oof_A3, oof_B, oof_C])
test_stack = np.column_stack([pred_A1, pred_A2, pred_A3, pred_B, pred_C])

# Pairwise correlation matrix
print("\n  Pairwise prediction correlations (lower = more diverse = better stack):")
corr_mat = np.corrcoef(oof_stack.T)
print(f"  {'':14s}" + "".join(f"{l:13s}" for l in labels))
for i, l in enumerate(labels):
    print(f"  {l:14s}" + "".join(f"{corr_mat[i,j]:13.4f}" for j in range(len(labels))))

# Ridge meta-learner
ridge = Ridge(alpha=10.0, fit_intercept=True)
ridge.fit(oof_stack, y_raw)
pred_meta = ridge.predict(test_stack)
oof_meta  = ridge.predict(oof_stack)
r2_meta   = r2_score(y_raw, oof_meta)

print(f"\n  Ridge coefficients:")
for name, coef in zip(labels, ridge.coef_):
    print(f"    {name:14s} → {coef:.4f}")
print(f"  Intercept: {ridge.intercept_:.0f}")
print(f"\n  [Ridge Stack] OOF R²: {r2_meta:.5f}")

# Optimized random blend as backup
print("\n  Running random blend search (8000 samples) …")
best_r2, best_ws = -99, None
np.random.seed(SEED)
for _ in range(8000):
    w  = np.random.dirichlet(np.ones(5))
    r2 = r2_score(y_raw, oof_stack @ w)
    if r2 > best_r2:
        best_r2, best_ws = r2, w.copy()
pred_blend = test_stack @ best_ws
r2_blend   = best_r2
print(f"  Best blend R²: {r2_blend:.5f}")
print(f"  Weights: {dict(zip(labels, [f'{x:.3f}' for x in best_ws]))}")

# Choose best
if r2_meta >= r2_blend:
    pred_final    = pred_meta
    final_label   = f"Ridge stack  R²={r2_meta:.5f}"
    best_final_r2 = r2_meta
else:
    pred_final    = pred_blend
    final_label   = f"Blend        R²={r2_blend:.5f}"
    best_final_r2 = r2_blend

print(f"\n  → Final: {final_label}")

# ─── POST-PROCESSING ──────────────────────────────────────────────────────────
pred_final = np.clip(pred_final, cltv_floor, cltv_true_max)
print(f"  Prediction range: [{pred_final.min():.0f}, {pred_final.max():.0f}]")

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
print("\n[Step] Feature importance (CatBoost A1) …")
fi_m = CatBoostRegressor(**{**CB_SYM, "iterations": 1500, "verbose": 0})
fi_m.fit(Pool(X, y_win, cat_features=cat_idx))

fi_df = (pd.DataFrame({"feature": ALL_FEATURES,
                        "importance": fi_m.get_feature_importance()})
           .sort_values("importance", ascending=False))

print("\n--- Feature Importances (all) ---")
print(fi_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 7))
sns.barplot(data=fi_df, x="importance", y="feature", palette="Blues_r", ax=ax)
ax.set_title("Feature Importance — v6 (21 clean features)")
plt.tight_layout()
plt.savefig("plots/feature_importance_v6.png", dpi=120)
plt.close()
print("  → plots/feature_importance_v6.png saved")

# ─── SUBMISSION ───────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "cltv": pred_final.astype(int)})
submission.to_csv("submission_v6.csv", index=False)
print(f"\n[Saved] submission_v6.csv  ({len(submission)} rows)")
print(submission.head(10).to_string(index=False))

# ─── SUMMARY ──────────────────────────────────────────────────────────────────
elapsed = (time.time() - START) / 60
print("\n" + "=" * 65)
print("  FINAL SUMMARY  (OOF R² on ORIGINAL CLTV)")
print("=" * 65)
print(f"  A1 — CB SymTree depth=6 RMSE   : {r2_A1:.5f}")
print(f"  A2 — CB Lossguide lv=64 RMSE   : {r2_A2:.5f}  [key: different tree shape]")
print(f"  A3 — CB SymTree depth=6 MAE    : {r2_A3:.5f}  [different loss]")
print(f"  B  — LightGBM lv=63 RMSE       : {r2_B:.5f}")
print(f"  C  — HistGB depth=6            : {r2_C:.5f}")
print(f"  ──────────────────────────────────────────────")
print(f"  Ridge stack                    : {r2_meta:.5f}")
print(f"  Optimized blend                : {r2_blend:.5f}")
print(f"  ══════════════════════════════════════════════")
print(f"  FINAL → {final_label}")
print(f"  Wall time: {elapsed:.1f} min")
print("=" * 65)
print("  submission_v6.csv ready  ✓")
print("=" * 65)     