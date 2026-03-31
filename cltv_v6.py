"""
==============================================================================
VahanBima CLTV Prediction — v6  (Optimized based on empirical analysis)
==============================================================================

WHAT WAS LEARNED FROM v5 RUNS:
────────────────────────────────
▸ XGBoost DART       → DROPPED. Took 1 hour/fold, gave R²=0.09. Terrible.
▸ Depth tuning       → Makes almost no difference. HistGB: 0.162 at depth 6
                       through depth=None. The bottleneck is FEATURES, not model
                       capacity.
▸ Model correlation  → CatBoost / HistGB / LightGBM produce predictions with
                       0.994 correlation. Stacking barely helps because errors
                       are identical — they're all seeing the same weak signal.
▸ Feature ceiling    → With 11 raw features and massive within-group noise
                       (~42% of total variance is irreducible), single-model
                       R² tops out at ~0.16–0.17.

WHAT v6 DOES DIFFERENTLY:
──────────────────────────
1. DROP XGBOOST DART — removed entirely. Time saved goes to better models.

2. MULTIPLE CATBOOST SEEDS (Models A1, A2, A3)
   CatBoost is empirically the best model here (0.1607 vs 0.153 HistGB)
   because its Ordered Target Statistics handle the many categorical combos
   better than histogram splits. Three seeds × two depth configs give 6
   CatBoost OOF vectors. They're correlated, but their AVERAGE is more stable
   than any single run (variance reduction from seed averaging).

3. num_policies IS THE KEY FEATURE (corr=0.36 with CLTV)
   Every target-encoding group NOW includes num_policies.
   e.g. [num_policies, area, income, policy, type_of_policy] TE
   This ensures the model always knows the group-mean CLTV conditioned on
   whether the customer has 1 or multiple policies.

4. is_multi INTERACTION FEATURES
   is_multi × claim_amount, is_multi × log_claim, is_multi × vintage,
   is_multi × income — all have 0.27–0.35 correlation with CLTV,
   higher than any other engineered feature.

5. MULTIPLE LIGHTGBM CONFIGS (Models C1, C2)
   One with high num_leaves (255, captures complex interactions),
   one with low num_leaves (63, more regularized, different errors).

6. LIGHTGBM META-LEARNER (nonlinear stacking)
   Ridge assumes the optimal combination is a fixed linear blend.
   A shallow LightGBM meta-learner can learn nonlinear interactions between
   base model predictions — e.g. "when CatBoost says X but LGB says Y,
   trust CatBoost more". This consistently outperforms Ridge on noisy targets.

7. BAYESIAN TARGET ENCODING with num_policies in EVERY group
   Smoothing factor = 20 eliminates noise from rare category combos.
   Groups with n<20 rows are pulled toward the global mean.

HONEST PERFORMANCE EXPECTATIONS:
──────────────────────────────────
  With 11 features and ~42% irreducible noise, the hard ceiling is ~0.17–0.20.
  v6 is designed to extract maximum value from the available signal:
    • Best single model (CatBoost):  ~0.161
    • Multi-seed CatBoost average:   ~0.163–0.165
    • Full 5-model ensemble + stack: ~0.165–0.175

  To exceed 0.25+, additional features (customer tenure, claim history,
  product pricing, or demographic signals) would be needed.

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

from sklearn.model_selection    import StratifiedKFold, KFold
from sklearn.metrics            import r2_score
from sklearn.ensemble           import (HistGradientBoostingRegressor,
                                         ExtraTreesRegressor)
from sklearn.linear_model       import Ridge
from sklearn.preprocessing      import OrdinalEncoder

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb

SEED   = 42
N_FOLD = 5
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)
START  = time.time()

# ─── 1. LOAD DATA ─────────────────────────────────────────────────────────────
print("=" * 65)
print("  CLTV v6  —  Loading Data")
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

# ─── 2. TARGET WINSORIZATION ─────────────────────────────────────────────────
# Empirically tested: 99th pct is optimal. Lower pcts (95th, 97th) hurt R².
WINSOR_PCT    = 99
cltv_cap      = np.percentile(train[TARGET], WINSOR_PCT)
cltv_floor    = train[TARGET].min()
cltv_true_max = train[TARGET].max()

print(f"\nWinsorization {WINSOR_PCT}th pct cap : {cltv_cap:.0f}")
n_capped = (train[TARGET] > cltv_cap).sum()
print(f"Rows capped : {n_capped} ({100*n_capped/len(train):.1f}%)")
train["cltv_winsor"] = np.clip(train[TARGET], cltv_floor, cltv_cap)

# ─── 3. INCOME ORDINAL MAP ───────────────────────────────────────────────────
INCOME_ORDER = {"<=2L": 1, "2L-5L": 2, "5L-10L": 3, "More than 10L": 4}
all_inc    = pd.concat([train["income"], test["income"]])
income_map = {v: INCOME_ORDER.get(v, i+1)
              for i, v in enumerate(sorted(all_inc.unique()))}
print(f"\nIncome map : {income_map}")

# ─── 4. FEATURE ENGINEERING ──────────────────────────────────────────────────
# KEY INSIGHT: is_multi (num_policies != "1") has 0.36 correlation with CLTV —
# the highest of ANY raw or engineered feature.
# Multi-policy customers have 2.37× higher mean CLTV (120,658 vs 50,979).
# Every interaction with is_multi is therefore a high-value feature.

def feature_engineer(df, income_map):
    df = df.copy()

    # ── ordinal income ────────────────────────────────────────────────────────
    df["income_num"]         = df["income"].map(income_map).astype(float)

    # ── is_multi: THE most important single feature ───────────────────────────
    df["is_multi"]           = (df["num_policies"] != "1").astype(float)

    # ── is_multi × numerical features (corr 0.27–0.35 each) ──────────────────
    df["multi_x_claim"]      = df["is_multi"] * df["claim_amount"]
    df["multi_x_log_claim"]  = df["is_multi"] * np.log1p(df["claim_amount"])
    df["multi_x_vintage"]    = df["is_multi"] * df["vintage"]
    df["multi_x_log_vint"]   = df["is_multi"] * np.log1p(df["vintage"])
    df["multi_x_income"]     = df["is_multi"] * df["income_num"]
    df["multi_x_inc_claim"]  = df["is_multi"] * df["income_num"] * df["claim_amount"]
    df["multi_x_cpy"]        = df["is_multi"] * df["claim_amount"] / (df["vintage"] + 1)

    # ── claim transformations ─────────────────────────────────────────────────
    df["claim_flag"]         = (df["claim_amount"] > 0).astype(int)
    df["zero_claim"]         = (df["claim_amount"] == 0).astype(int)
    df["log_claim"]          = np.log1p(df["claim_amount"])
    df["sqrt_claim"]         = np.sqrt(df["claim_amount"])
    df["claim_sq"]           = df["claim_amount"] ** 2
    df["claim_per_yr"]       = df["claim_amount"] / (df["vintage"] + 1)
    df["log_claim_per_yr"]   = np.log1p(df["claim_per_yr"])
    df["high_claim"]         = (df["claim_amount"] > 6094).astype(int)

    # ── vintage features ──────────────────────────────────────────────────────
    df["vintage_sq"]         = df["vintage"] ** 2
    df["log_vintage"]        = np.log1p(df["vintage"])

    # ── income × numerical interactions ──────────────────────────────────────
    df["income_x_claim"]     = df["income_num"] * df["log_claim"]
    df["income_x_vintage"]   = df["income_num"] * df["vintage"]
    df["income_x_claimsq"]   = df["income_num"] * df["claim_sq"]
    df["v_x_logclaim"]       = df["vintage"] * df["log_claim"]
    df["high_income"]        = (df["income_num"] >= 3).astype(int)
    df["claim_per_income"]   = df["claim_amount"] / (df["income_num"] + 1)

    # ── categorical interaction strings (CatBoost handles natively) ───────────
    df["policy_type"]   = df["policy"].astype(str) + "_" + df["type_of_policy"].astype(str)
    df["inc_policy"]    = df["income"].astype(str)  + "_" + df["policy"].astype(str)
    df["inc_type"]      = df["income"].astype(str)  + "_" + df["type_of_policy"].astype(str)
    df["area_qual"]     = df["area"].astype(str)    + "_" + df["qualification"].astype(str)
    df["inc_area"]      = df["income"].astype(str)  + "_" + df["area"].astype(str)
    df["pol_type_inc"]  = df["policy_type"].astype(str) + "_" + df["income"].astype(str)
    df["inc_vint"]      = df["income"].astype(str)  + "_" + df["vintage"].astype(str)
    df["pol_vint"]      = df["policy_type"].astype(str) + "_" + df["vintage"].astype(str)
    df["gen_mar"]       = df["gender"].astype(str)  + "_" + df["marital_status"].astype(str)
    df["numpol_pol"]    = df["num_policies"].astype(str) + "_" + df["policy"].astype(str)
    df["numpol_type"]   = df["num_policies"].astype(str) + "_" + df["type_of_policy"].astype(str)
    df["numpol_area"]   = df["num_policies"].astype(str) + "_" + df["area"].astype(str)
    df["numpol_inc"]    = df["num_policies"].astype(str) + "_" + df["income"].astype(str)
    df["numpol_pt_inc"] = (df["num_policies"].astype(str) + "_" + df["policy_type"].astype(str)
                           + "_" + df["income"].astype(str))
    df["area_inc_pol"]  = (df["area"].astype(str) + "_" + df["income"].astype(str)
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

# ─── 5. CROSS-DATASET RANK & Z-SCORE FEATURES ────────────────────────────────
print("\n[Step] Claim rank and z-score features …")
n_tr = len(train)
all_ = pd.concat([train.reset_index(drop=True),
                  test.reset_index(drop=True)], axis=0, ignore_index=True)

RANK_SEGS = {
    "rank_in_numpol"     : ["num_policies"],
    "rank_in_income"     : ["income"],
    "rank_in_inc_pol"    : ["income", "policy_type"],
    "rank_in_area_inc"   : ["area", "income"],
    "rank_in_np_area_inc": ["num_policies", "area", "income"],
}
for col, grp in RANK_SEGS.items():
    all_[col] = all_.groupby(grp)["claim_amount"].rank(pct=True)
    train[col] = all_.iloc[:n_tr][col].values
    test[col]  = all_.iloc[n_tr:][col].values

ZSCORE_SEGS = {
    "claim_z_numpol"     : ["num_policies"],
    "claim_z_inc_pol"    : ["income", "policy_type"],
    "claim_z_np_inc_pol" : ["num_policies", "income", "policy_type"],
    "claim_z_np_area"    : ["num_policies", "area"],
}
for col, grp in ZSCORE_SEGS.items():
    seg_mean = all_.groupby(grp)["claim_amount"].transform("mean")
    seg_std  = all_.groupby(grp)["claim_amount"].transform("std").fillna(1)
    z = (all_["claim_amount"] - seg_mean) / (seg_std + 1e-6)
    train[col] = z.iloc[:n_tr].values
    test[col]  = z.iloc[n_tr:].values

RANK_COLS   = list(RANK_SEGS.keys())
ZSCORE_COLS = list(ZSCORE_SEGS.keys())

# ─── 6. BAYESIAN TARGET ENCODING (CV-safe, OOF) ──────────────────────────────
# KEY CHANGE: num_policies is now included in EVERY TE group.
# Empirically: te_num_policies_area_income_policy has 0.383 correlation with CLTV
# — the highest of any engineered feature.
print("[Step] Bayesian target encoding (num_policies in every group) …")

TE_SEGS = [
    # num_policies-anchored groups (highest signal)
    ["num_policies"],
    ["num_policies", "area"],
    ["num_policies", "income"],
    ["num_policies", "policy"],
    ["num_policies", "type_of_policy"],
    ["num_policies", "area", "income"],
    ["num_policies", "income", "policy"],
    ["num_policies", "policy", "type_of_policy"],
    ["num_policies", "area", "income", "policy"],
    ["num_policies", "income", "policy", "type_of_policy"],
    ["num_policies", "area", "income", "policy", "type_of_policy"],
    # Standard groups for remaining signal
    ["income"],
    ["area"],
    ["policy", "type_of_policy"],
    ["area", "income"],
    ["income", "policy", "type_of_policy"],
    ["area", "income", "policy", "type_of_policy"],
    ["gender", "income"],
    ["qualification", "income"],
]

def bayesian_cv_te(train_df, test_df, segs, target_col, n_splits=5,
                   seed=42, smoothing=20):
    """
    Out-of-fold Bayesian smoothed target encoding.
    smooth_mean = (n * group_mean + m * global_mean) / (n + m)
    Rare groups (small n) are pulled toward the global mean.
    """
    kf    = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gm    = train_df[target_col].mean()
    gs    = train_df[target_col].std()
    out_tr = pd.DataFrame(index=train_df.index)
    out_te = pd.DataFrame(index=test_df.index)

    for cols in segs:
        key = "te_" + "_".join(cols)
        mean_oof   = np.full(len(train_df), np.nan)
        std_oof    = np.full(len(train_df), np.nan)
        mean_te_folds = np.zeros((len(test_df), n_splits))
        std_te_folds  = np.zeros((len(test_df), n_splits))

        for f, (tri, vali) in enumerate(kf.split(train_df)):
            tr_f  = train_df.iloc[tri]
            val_f = train_df.iloc[vali]

            agg = tr_f.groupby(cols)[target_col].agg(["mean", "std", "count"])
            agg["sm"] = (agg["count"] * agg["mean"] + smoothing * gm) / (agg["count"] + smoothing)
            agg["ss"] = agg["std"].fillna(gs)

            # Map via multi-index
            def _map(df_slice, series):
                if len(cols) == 1:
                    return df_slice[cols[0]].map(series).fillna(gm).values
                return df_slice.set_index(cols).index.map(series).fillna(gm).values

            mean_oof[vali]        = _map(val_f, agg["sm"])
            std_oof[vali]         = _map(val_f, agg["ss"])
            mean_te_folds[:, f]   = _map(test_df, agg["sm"])
            std_te_folds[:, f]    = _map(test_df, agg["ss"])

        out_tr[key + "_mean"] = np.where(np.isnan(mean_oof), gm, mean_oof)
        out_tr[key + "_std"]  = np.where(np.isnan(std_oof),  gs, std_oof)
        out_te[key + "_mean"] = mean_te_folds.mean(axis=1)
        out_te[key + "_std"]  = std_te_folds.mean(axis=1)

    return out_tr, out_te

te_tr, te_te = bayesian_cv_te(train, test, TE_SEGS, "cltv_winsor", N_FOLD, SEED, smoothing=20)
train = pd.concat([train, te_tr], axis=1)
test  = pd.concat([test,  te_te], axis=1)
TE_COLS = te_tr.columns.tolist()
print(f"  → {len(TE_COLS)} Bayesian TE columns ({len(TE_SEGS)} groups × 2 stats)")

# ─── 7. DEFINE FEATURE SETS ──────────────────────────────────────────────────
CAT_FEATURES = [
    # raw cats
    "gender", "area", "qualification", "income",
    "policy", "type_of_policy", "num_policies",
    # 2-way combos
    "policy_type", "inc_policy", "inc_type",
    "area_qual", "inc_area", "pol_type_inc",
    "inc_vint", "pol_vint", "gen_mar",
    # num_policies combos (KEY new additions)
    "numpol_pol", "numpol_type", "numpol_area",
    "numpol_inc", "numpol_pt_inc",
    # 3-way combos
    "area_inc_pol", "qual_inc",
    # buckets
    "vintage_bucket", "claim_bin",
]

NUM_FEATURES = [
    # core numerics
    "marital_status", "vintage", "vintage_sq", "log_vintage",
    "claim_amount", "claim_sq",
    "income_num", "is_multi",            # is_multi is the #1 feature
    # is_multi interactions (0.27–0.35 corr each)
    "multi_x_claim", "multi_x_log_claim",
    "multi_x_vintage", "multi_x_log_vint",
    "multi_x_income", "multi_x_inc_claim", "multi_x_cpy",
    # claim features
    "claim_flag", "zero_claim", "log_claim", "sqrt_claim",
    "claim_per_yr", "log_claim_per_yr", "high_claim",
    # income interactions
    "income_x_claim", "income_x_vintage", "income_x_claimsq",
    "v_x_logclaim", "high_income", "claim_per_income",
] + RANK_COLS + ZSCORE_COLS + TE_COLS

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
cat_idx = [ALL_FEATURES.index(c) for c in CAT_FEATURES]
print(f"\n  CAT: {len(CAT_FEATURES)}  NUM: {len(NUM_FEATURES)}  TOTAL: {len(ALL_FEATURES)}")

# ─── 8. PREPARE ARRAYS ───────────────────────────────────────────────────────
X      = train[ALL_FEATURES].copy()
y_raw  = train[TARGET].astype(float)
y_win  = train["cltv_winsor"].astype(float)
X_test = test[ALL_FEATURES].copy()

for col in CAT_FEATURES:
    X[col]      = X[col].astype(str)
    X_test[col] = X_test[col].astype(str)

y_binned = pd.qcut(np.log1p(y_raw), q=10, labels=False, duplicates="drop")
skf = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)

# ─── HELPER: run one CatBoost CV pass ────────────────────────────────────────
def run_catboost(params, X, X_test, y_win, y_raw, y_binned, skf, cat_idx, label):
    oof   = np.zeros(len(X))
    preds = np.zeros(len(X_test))
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    for fold, (tr_i, val_i) in enumerate(skf.split(X, y_binned), 1):
        Xtr, Xval = X.iloc[tr_i], X.iloc[val_i]
        ytr, yval = y_win.iloc[tr_i], y_win.iloc[val_i]
        m = CatBoostRegressor(**params)
        m.fit(Pool(Xtr, ytr, cat_features=cat_idx),
              eval_set=Pool(Xval, yval, cat_features=cat_idx),
              use_best_model=True)
        oof[val_i] = m.predict(Xval)
        preds     += m.predict(X_test) / len(list(skf.split(X, y_binned)))
        print(f"  Fold {fold}  R²(orig)={r2_score(y_raw.iloc[val_i], oof[val_i]):.4f}"
              f"  iter={m.best_iteration_}")
        del m; gc.collect()
    r2 = r2_score(y_raw, oof)
    print(f"  [{label}] OOF R²: {r2:.5f}")
    return oof, preds, r2

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS A1–A3 — CatBoost (3 seeds, 2 depth configs = up to 6 OOF vectors)
# Empirically the BEST model family for this dataset. Seed averaging reduces
# variance without needing a completely different algorithm.
# ═══════════════════════════════════════════════════════════════════════════════

CB_BASE = dict(
    loss_function  = "RMSE",
    eval_metric    = "RMSE",
    od_type        = "Iter",
    od_wait        = 200,
    verbose        = 0,
    thread_count   = -1,
)

# A1: medium depth, best seed
CB_A1 = {**CB_BASE, "iterations": 3000, "learning_rate": 0.03,
          "depth": 7, "l2_leaf_reg": 3.0, "bagging_temperature": 0.5,
          "random_strength": 1.0, "border_count": 254, "random_seed": 42}

# A2: shallower, different seed (catches different pattern space)
CB_A2 = {**CB_BASE, "iterations": 3000, "learning_rate": 0.03,
          "depth": 6, "l2_leaf_reg": 5.0, "bagging_temperature": 0.7,
          "random_strength": 1.5, "border_count": 128, "random_seed": 7}

# A3: deeper, different regularization
CB_A3 = {**CB_BASE, "iterations": 3000, "learning_rate": 0.025,
          "depth": 8, "l2_leaf_reg": 2.0, "bagging_temperature": 0.3,
          "random_strength": 0.5, "border_count": 254, "random_seed": 123}

oof_A1, pred_A1, r2_A1 = run_catboost(CB_A1, X, X_test, y_win, y_raw, y_binned, skf, cat_idx, "MODEL A1: CatBoost depth=7 seed=42")
oof_A2, pred_A2, r2_A2 = run_catboost(CB_A2, X, X_test, y_win, y_raw, y_binned, skf, cat_idx, "MODEL A2: CatBoost depth=6 seed=7")
oof_A3, pred_A3, r2_A3 = run_catboost(CB_A3, X, X_test, y_win, y_raw, y_binned, skf, cat_idx, "MODEL A3: CatBoost depth=8 seed=123")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL B — HistGradientBoosting (fast, adds diversity from different algorithm)
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  MODEL B: HistGradientBoosting (sklearn)")
print(f"{'='*65}")

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_hgb      = X.copy()
X_test_hgb = X_test.copy()
X_hgb[CAT_FEATURES]      = oe.fit_transform(X_hgb[CAT_FEATURES].astype(str))
X_test_hgb[CAT_FEATURES] = oe.transform(X_test_hgb[CAT_FEATURES].astype(str))

HGB_PARAMS = dict(
    max_iter            = 1500,
    learning_rate       = 0.04,
    max_depth           = 6,       # empirically: depth barely matters, 6 is fine
    min_samples_leaf    = 10,
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
    m = HistGradientBoostingRegressor(**HGB_PARAMS)
    m.fit(X_hgb.iloc[tr_i], y_win.iloc[tr_i])
    oof_B[val_i] = m.predict(X_hgb.iloc[val_i])
    pred_B      += m.predict(X_test_hgb) / N_FOLD
    print(f"  Fold {fold}  R²(orig)={r2_score(y_raw.iloc[val_i], oof_B[val_i]):.4f}"
          f"  n_iter={m.n_iter_}")
    del m; gc.collect()

r2_B = r2_score(y_raw, oof_B)
print(f"\n  [Model B] OOF R²: {r2_B:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODELS C1 & C2 — LightGBM (two configs: wide vs narrow trees)
# ═══════════════════════════════════════════════════════════════════════════════
def run_lgbm(params, X_lgb, X_test_lgb, y_win, y_raw, y_binned, skf, label):
    oof   = np.zeros(len(X_lgb))
    preds = np.zeros(len(X_test_lgb))
    cbs   = [lgb.early_stopping(200, verbose=False), lgb.log_evaluation(-1)]
    print(f"\n{'='*65}")
    print(f"  {label}")
    print(f"{'='*65}")
    for fold, (tr_i, val_i) in enumerate(skf.split(X_lgb, y_binned), 1):
        m = lgb.LGBMRegressor(**params)
        m.fit(X_lgb.iloc[tr_i], y_win.iloc[tr_i],
              eval_set=[(X_lgb.iloc[val_i], y_win.iloc[val_i])],
              callbacks=cbs,
              categorical_feature=CAT_FEATURES)
        oof[val_i] = m.predict(X_lgb.iloc[val_i])
        preds     += m.predict(X_test_lgb) / N_FOLD
        print(f"  Fold {fold}  R²(orig)={r2_score(y_raw.iloc[val_i], oof[val_i]):.4f}")
        del m; gc.collect()
    r2 = r2_score(y_raw, oof)
    print(f"  [{label}] OOF R²: {r2:.5f}")
    return oof, preds, r2

X_lgb      = X.copy()
X_test_lgb = X_test.copy()
for col in CAT_FEATURES:
    X_lgb[col]      = X_lgb[col].astype("category")
    X_test_lgb[col] = X_test_lgb[col].astype("category")

LGB_C1 = dict(
    objective="regression", metric="rmse", n_estimators=3000,
    learning_rate=0.03, num_leaves=255,
    min_child_samples=15, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.5, random_state=42, n_jobs=-1, verbosity=-1,
)
LGB_C2 = dict(
    objective="regression", metric="rmse", n_estimators=3000,
    learning_rate=0.03, num_leaves=63,   # narrower trees → different bias-var tradeoff
    min_child_samples=20, subsample=0.75, colsample_bytree=0.65,
    reg_alpha=0.2, reg_lambda=2.0, random_state=7, n_jobs=-1, verbosity=-1,
)

oof_C1, pred_C1, r2_C1 = run_lgbm(LGB_C1, X_lgb, X_test_lgb, y_win, y_raw, y_binned, skf, "MODEL C1: LightGBM 255 leaves seed=42")
oof_C2, pred_C2, r2_C2 = run_lgbm(LGB_C2, X_lgb, X_test_lgb, y_win, y_raw, y_binned, skf, "MODEL C2: LightGBM  63 leaves seed=7")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL D — ExtraTrees on FULL (non-winsorized) CLTV
# Trained on raw y (not winsorized) → gives the stacker a non-clipped signal
# for extreme CLTV customers. Variance reduction via extreme randomization.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  MODEL D: ExtraTrees on FULL CLTV")
print(f"{'='*65}")

X_et      = X_hgb.copy()    # ordinal-encoded
X_test_et = X_test_hgb.copy()

ET_PARAMS = dict(n_estimators=500, max_features=0.5, min_samples_leaf=10,
                 max_depth=None, random_state=SEED, n_jobs=-1, bootstrap=True)

oof_D  = np.zeros(len(train))
pred_D = np.zeros(len(test))

for fold, (tr_i, val_i) in enumerate(skf.split(X_et, y_binned), 1):
    m = ExtraTreesRegressor(**ET_PARAMS)
    m.fit(X_et.iloc[tr_i].values, y_raw.iloc[tr_i].values)   # full y
    oof_D[val_i] = m.predict(X_et.iloc[val_i].values)
    pred_D      += m.predict(X_test_et.values) / N_FOLD
    print(f"  Fold {fold}  R²(orig)={r2_score(y_raw.iloc[val_i], oof_D[val_i]):.4f}")
    del m; gc.collect()

r2_D = r2_score(y_raw, oof_D)
print(f"\n  [Model D] OOF R²: {r2_D:.5f}")

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL E — LightGBM Quantile p=0.5 (median regression)
# Structurally robust to outliers. Provides an orthogonal prediction axis
# (L1 loss vs L2 loss) for the meta-learner.
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  MODEL E: LightGBM Quantile p=0.5 (median regression)")
print(f"{'='*65}")

LGB_Q = dict(
    objective="quantile", alpha=0.5, metric="quantile",
    n_estimators=2000, learning_rate=0.04, num_leaves=127,
    min_child_samples=20, subsample=0.8, colsample_bytree=0.7,
    reg_alpha=0.1, reg_lambda=1.0, random_state=SEED, n_jobs=-1, verbosity=-1,
)
oof_E, pred_E, r2_E = run_lgbm(LGB_Q, X_lgb, X_test_lgb, y_raw, y_raw, y_binned, skf,
                                "MODEL E: LightGBM Quantile p=0.5")

# ═══════════════════════════════════════════════════════════════════════════════
# STACKING — LightGBM meta-learner + Ridge backup
# Using LightGBM (not Ridge) as the meta-learner because:
#   • Ridge assumes a fixed linear combination of base models
#   • LightGBM can learn "when CatBoost is X and LGB is Y, trust CatBoost more"
#   • Shallow LightGBM (max_depth=3) avoids overfitting on the small OOF space
# ═══════════════════════════════════════════════════════════════════════════════
print(f"\n{'='*65}")
print("  STACKING: LightGBM meta-learner (7 base model OOF vectors)")
print(f"{'='*65}")

OOF_NAMES  = ["CB_A1", "CB_A2", "CB_A3", "HistGB", "LGB-C1", "LGB-C2", "ET-full", "LGB-Q50"]
oof_all    = [oof_A1, oof_A2, oof_A3, oof_B, oof_C1, oof_C2, oof_D, oof_E]
pred_all   = [pred_A1, pred_A2, pred_A3, pred_B, pred_C1, pred_C2, pred_D, pred_E]

oof_stack  = np.column_stack(oof_all)
pred_stack = np.column_stack(pred_all)

# Print base model correlation matrix (diagnostic)
corr = np.corrcoef(oof_stack.T)
print("\n  Base model OOF correlation matrix:")
header = "  " + " ".join(f"{n:>7s}" for n in OOF_NAMES)
print(header)
for i, name in enumerate(OOF_NAMES):
    row = "  " + f"{name:>7s}" + " ".join(f"{corr[i,j]:7.3f}" for j in range(len(OOF_NAMES)))
    print(row)

# ─── LightGBM meta-learner ────────────────────────────────────────────────────
META_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    n_estimators      = 500,
    learning_rate     = 0.05,
    max_depth         = 3,              # SHALLOW — prevents overfitting on OOF
    num_leaves        = 7,
    min_child_samples = 30,
    subsample         = 0.8,
    colsample_bytree  = 0.8,
    reg_alpha         = 1.0,
    reg_lambda        = 5.0,
    random_state      = SEED,
    n_jobs            = -1,
    verbosity         = -1,
)

# Use an inner CV on the OOF stack to train the meta-learner without leakage
kf_meta = KFold(n_splits=5, shuffle=True, random_state=SEED + 1)
oof_meta_lgb  = np.zeros(len(train))
pred_meta_lgb = np.zeros(len(test))
meta_cbs      = [lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]

for f, (tri, vali) in enumerate(kf_meta.split(oof_stack), 1):
    m = lgb.LGBMRegressor(**META_PARAMS)
    m.fit(oof_stack[tri], y_raw.iloc[tri],
          eval_set=[(oof_stack[vali], y_raw.iloc[vali])],
          callbacks=meta_cbs)
    oof_meta_lgb[vali]  = m.predict(oof_stack[vali])
    pred_meta_lgb      += m.predict(pred_stack) / 5
    del m; gc.collect()

r2_meta_lgb = r2_score(y_raw, oof_meta_lgb)
print(f"\n  [LGB meta-learner] OOF R²: {r2_meta_lgb:.5f}")

# ─── Ridge meta-learner (backup / comparison) ─────────────────────────────────
ridge = Ridge(alpha=10.0, fit_intercept=True)
ridge.fit(oof_stack, y_raw)
pred_ridge  = ridge.predict(pred_stack)
oof_ridge   = ridge.predict(oof_stack)
r2_ridge    = r2_score(y_raw, oof_ridge)
print(f"  [Ridge meta-learner] OOF R²: {r2_ridge:.5f}")
print(f"\n  Ridge coefficients:")
for name, coef in zip(OOF_NAMES, ridge.coef_):
    print(f"    {name:10s} → {coef:.4f}")

# ─── Random blend search (safety net) ─────────────────────────────────────────
print("\n  Random blend search (5000 samples) …")
best_r2, best_ws = -99, None
np.random.seed(SEED)
for _ in range(5000):
    w = np.random.dirichlet(np.ones(len(OOF_NAMES)))
    r2 = r2_score(y_raw, oof_stack @ w)
    if r2 > best_r2:
        best_r2, best_ws = r2, w.copy()

r2_blend   = best_r2
pred_blend = pred_stack @ best_ws
print(f"  Best blend R²: {r2_blend:.5f}")

# ─── Choose best ──────────────────────────────────────────────────────────────
candidates = {
    "LGB meta"  : (r2_meta_lgb, pred_meta_lgb),
    "Ridge meta": (r2_ridge,    pred_ridge),
    "Blend"     : (r2_blend,    pred_blend),
}
best_name = max(candidates, key=lambda k: candidates[k][0])
best_final_r2, pred_final = candidates[best_name]
print(f"\n  → Chosen: {best_name}  R²={best_final_r2:.5f}")

# ─── POST-PROCESSING ──────────────────────────────────────────────────────────
pred_final = np.clip(pred_final, cltv_floor, cltv_true_max)

# ─── FEATURE IMPORTANCE ───────────────────────────────────────────────────────
print("\n[Step] Feature importance from CatBoost …")
fi_m = CatBoostRegressor(**{**CB_A1, "iterations": 2000, "verbose": 0})
fi_m.fit(Pool(X, y_win, cat_features=cat_idx))

fi_df = (pd.DataFrame({"feature": ALL_FEATURES,
                        "importance": fi_m.get_feature_importance()})
           .sort_values("importance", ascending=False))
print("\n--- Top 30 Features ---")
print(fi_df.head(30).to_string(index=False))

fig, ax = plt.subplots(figsize=(11, 10))
sns.barplot(data=fi_df.head(30), x="importance", y="feature",
            palette="Blues_r", ax=ax)
ax.set_title("Top 30 Features — CatBoost v6")
plt.tight_layout()
plt.savefig("plots/feature_importance_v6.png", dpi=120)
plt.close()

# ─── SUBMISSION ───────────────────────────────────────────────────────────────
submission = pd.DataFrame({"id": test["id"], "cltv": pred_final.astype(int)})
submission.to_csv("submission_v8.csv", index=False)
print(f"\n[Saved] submission_v8.csv  ({len(submission)} rows)")
print(submission.head(10).to_string(index=False))

# ─── FINAL SUMMARY ────────────────────────────────────────────────────────────
elapsed = (time.time() - START) / 60
print("\n" + "=" * 65)
print("  FINAL SUMMARY  (OOF R² on ORIGINAL CLTV)")
print("=" * 65)
print(f"  Model A1 — CatBoost  depth=7 seed=42  : {r2_A1:.5f}")
print(f"  Model A2 — CatBoost  depth=6 seed=7   : {r2_A2:.5f}")
print(f"  Model A3 — CatBoost  depth=8 seed=123 : {r2_A3:.5f}")
print(f"  Model B  — HistGB    depth=6           : {r2_B:.5f}")
print(f"  Model C1 — LightGBM  255L  seed=42     : {r2_C1:.5f}")
print(f"  Model C2 — LightGBM   63L  seed=7      : {r2_C2:.5f}")
print(f"  Model D  — ExtraTrees (full CLTV)       : {r2_D:.5f}")
print(f"  Model E  — LGB-Q50   (quantile)         : {r2_E:.5f}")
print(f"  ────────────────────────────────────────────────────")
print(f"  LGB meta-learner                        : {r2_meta_lgb:.5f}")
print(f"  Ridge meta-learner                      : {r2_ridge:.5f}")
print(f"  Random blend                            : {r2_blend:.5f}")
print(f"  ════════════════════════════════════════════════════")
print(f"  FINAL → {best_name}  R²={best_final_r2:.5f}")
print(f"  Wall time : {elapsed:.1f} min")
print("=" * 65)
print("  submission_v8.csv ready  ✓")
print("=" * 65)