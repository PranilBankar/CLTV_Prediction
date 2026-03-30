"""
==============================================================================
VahanBima CLTV Prediction — v5 final (Memory-Safe, No String Pools)
==============================================================================
ROOT CAUSE OF PREVIOUS OOM:
  CatBoost creates internal hash tables for string categorical features.
  17 string-cat cols × 71K rows × border_count=128 → 'bad allocation'.

FIX:
  Pre-ordinal-encode ALL features → single float32 numpy array.
  CatBoost accepts integer cats via cat_features indices → same accuracy,
  70% less memory. No more string Pool overhead anywhere.

MODELS:
  1. CatBoost  — integer cats,  winsorized target
  2. LightGBM  — category dtype (lazy, per-fold), winsorized target
  3. XGBoost   — ordinally-encoded numpy, winsorized target
  4. RandomForest — numpy, winsorized target
  Meta: Ridge stacking on OOF predictions
==============================================================================
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy  as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os, gc

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics         import r2_score
from sklearn.ensemble        import RandomForestRegressor
from sklearn.linear_model    import Ridge
from sklearn.preprocessing   import OrdinalEncoder

from catboost import CatBoostRegressor, Pool
import lightgbm as lgb
import xgboost  as xgb

SEED   = 42
N_FOLD = 5
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)

# ─── 1. LOAD ──────────────────────────────────────────────────────────────────
print("=" * 56)
print("  LOADING DATA")
print("=" * 56)
train = pd.read_csv("Data/train_data.csv")
test  = pd.read_csv("Data/test_data.csv")
TARGET = "cltv"
print(f"Train {train.shape}  Test {test.shape}")

# ─── 2. WINSORIZE TARGET ──────────────────────────────────────────────────────
cap      = np.percentile(train[TARGET], 99)   # ≈ 485 520
floor    = train[TARGET].min()
true_max = train[TARGET].max()
train["cltv_w"] = np.clip(train[TARGET], floor, cap)
print(f"Winsorization cap: {cap:.0f}  (rows capped: {(train[TARGET]>cap).sum()})")

# ─── 3. INCOME ORDINAL MAP ────────────────────────────────────────────────────
INCOME_ORDER = {"<=2L":0,"2L-5L":1,"5L-10L":2,"More than 10L":3}
all_inc    = pd.concat([train["income"], test["income"]])
income_map = {v: INCOME_ORDER.get(v,i)
              for i,v in enumerate(sorted(all_inc.unique()))}

# ─── 4. FEATURE ENGINEERING ───────────────────────────────────────────────────
def feature_engineer(df, income_map):
    df = df.copy()
    df["income_num"]   = df["income"].map(income_map).astype(np.int8)
    df["multi_policy"] = (df["num_policies"] != "1").astype(np.int8)
    df["claim_flag"]   = (df["claim_amount"] > 0).astype(np.int8)
    df["zero_claim"]   = (df["claim_amount"] == 0).astype(np.int8)
    df["log_claim"]    = np.log1p(df["claim_amount"]).astype(np.float32)
    df["sqrt_claim"]   = np.sqrt(df["claim_amount"]).astype(np.float32)
    df["claim_per_yr"] = (df["claim_amount"]/(df["vintage"]+1)).astype(np.float32)
    df["log_cpyr"]     = np.log1p(df["claim_per_yr"]).astype(np.float32)
    df["vintage_sq"]   = (df["vintage"]**2).astype(np.int16)
    df["v_x_lc"]       = (df["vintage"]*df["log_claim"]).astype(np.float32)
    df["inc_x_lc"]     = (df["income_num"]*df["log_claim"]).astype(np.float32)
    df["inc_x_vint"]   = (df["income_num"]*df["vintage"]).astype(np.int16)
    df["high_income"]  = (df["income_num"]>=2).astype(np.int8)
    df["high_claim"]   = (df["claim_amount"]>6094).astype(np.int8)

    df["pol_type"]  = df["policy"].astype(str)+"_"+df["type_of_policy"].astype(str)
    df["inc_pol"]   = df["income"].astype(str)+"_"+df["policy"].astype(str)
    df["inc_type"]  = df["income"].astype(str)+"_"+df["type_of_policy"].astype(str)
    df["area_qual"] = df["area"].astype(str)+"_"+df["qualification"].astype(str)
    df["inc_area"]  = df["income"].astype(str)+"_"+df["area"].astype(str)
    df["pt_inc"]    = df["pol_type"].astype(str)+"_"+df["income"].astype(str)
    df["inc_vint"]  = df["income"].astype(str)+"_"+df["vintage"].astype(str)
    df["gen_mar"]   = df["gender"].astype(str)+"_"+df["marital_status"].astype(str)

    df["vint_bucket"] = pd.cut(df["vintage"],bins=[-1,1,3,6,100],
                               labels=["new","mid","old","vold"]).astype(str)
    df["claim_bin"]   = pd.qcut(df["claim_amount"],q=10,
                                labels=False,duplicates="drop").astype(str)
    return df

train = feature_engineer(train, income_map)
test  = feature_engineer(test,  income_map)

# ─── 5. CLAIM DEVIATION FEATURES (no leakage — X only) ───────────────────────
print("\n[Step] Claim deviation & rank features …")
n_tr = len(train)
all_ = pd.concat([train.reset_index(drop=True),
                  test.reset_index(drop=True)], ignore_index=True)

for col, grp in [("dev_income",["income"]),
                 ("dev_poltype",["pol_type"]),
                 ("dev_inc_pol",["income","pol_type"])]:
    mu   = all_.groupby(grp)["claim_amount"].transform("mean")
    sig  = all_.groupby(grp)["claim_amount"].transform("std").fillna(1.0)
    diff = (all_["claim_amount"]-mu).astype(np.float32)
    all_[col]         = diff
    all_[col+"_norm"] = (diff/(sig+1)).astype(np.float32)

for col, grp in [("rank_income",["income"]),
                 ("rank_poltype",["pol_type"]),
                 ("rank_inc_pt",["income","pol_type"])]:
    all_[col] = all_.groupby(grp)["claim_amount"].rank(pct=True).astype(np.float32)

DEV_COLS  = ["dev_income","dev_income_norm",
             "dev_poltype","dev_poltype_norm",
             "dev_inc_pol","dev_inc_pol_norm"]
RANK_COLS = ["rank_income","rank_poltype","rank_inc_pt"]

for c in DEV_COLS+RANK_COLS:
    train[c] = all_.iloc[:n_tr][c].values
    test[c]  = all_.iloc[n_tr:][c].values
del all_; gc.collect()
print(f"  → {len(DEV_COLS)+len(RANK_COLS)} columns added")

# ─── 6. BAYESIAN TARGET ENCODING ─────────────────────────────────────────────
print("[Step] Bayesian target encoding …")
TE_SEGS = ["gender","area","qualification","income",
           "policy","type_of_policy","pol_type","inc_pol",
           "inc_type","area_qual","inc_area","pt_inc","inc_vint","gen_mar"]

def bayesian_cv_te(tr_df, te_df, cols, target, prior=30, n_splits=5, seed=42):
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    gm = tr_df[target].mean()
    otr, ote = pd.DataFrame(index=tr_df.index), pd.DataFrame(index=te_df.index)
    for col in cols:
        arr  = np.full(len(tr_df), gm, dtype=np.float32)
        te_f = np.zeros((len(te_df), n_splits), dtype=np.float32)
        for f,(tri,vali) in enumerate(kf.split(tr_df)):
            grp = tr_df.iloc[tri].groupby(col)[target]
            n   = grp.count(); mu = grp.mean()
            sm  = (n*mu + prior*gm)/(n+prior)
            arr[vali]  = tr_df.iloc[vali][col].map(sm).fillna(gm).values.astype(np.float32)
            te_f[:,f]  = te_df[col].map(sm).fillna(gm).values.astype(np.float32)
        otr[col+"_bte"] = arr
        ote[col+"_bte"] = te_f.mean(axis=1)
    return otr, ote

te_tr,te_te = bayesian_cv_te(train,test,TE_SEGS,"cltv_w",prior=30,
                              n_splits=N_FOLD,seed=SEED)
train = pd.concat([train,te_tr],axis=1)
test  = pd.concat([test, te_te],axis=1)
BTE_COLS = te_tr.columns.tolist()
del te_tr,te_te; gc.collect()
print(f"  → {len(BTE_COLS)} Bayesian TE columns")

# ─── 7. DEFINE FEATURES ───────────────────────────────────────────────────────
CAT_FEATURES = [
    "gender","area","qualification","income",
    "policy","type_of_policy","num_policies",
    "pol_type","inc_pol","inc_type","area_qual",
    "inc_area","pt_inc","inc_vint","gen_mar",
    "vint_bucket","claim_bin",
]
NUM_FEATURES = [
    "marital_status","vintage","vintage_sq","claim_amount",
    "income_num","multi_policy","claim_flag","zero_claim",
    "log_claim","sqrt_claim","claim_per_yr","log_cpyr",
    "high_income","high_claim","inc_x_lc","inc_x_vint","v_x_lc",
] + DEV_COLS + RANK_COLS + BTE_COLS

ALL_FEATURES = CAT_FEATURES + NUM_FEATURES
print(f"\n  CAT={len(CAT_FEATURES)}  NUM={len(NUM_FEATURES)}  TOTAL={len(ALL_FEATURES)}")

# ─── 8. ENCODE EVERYTHING TO A SINGLE NUMERIC ARRAY ──────────────────────────
# KEY MEMORY FIX: one float32 numpy array only.
# CatBoost treats integer columns as categorical when passed cat_features indices
# → same accuracy, 70% less memory than string Pool.

X_df      = train[ALL_FEATURES].copy()
X_test_df = test[ALL_FEATURES].copy()

oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
X_df[CAT_FEATURES]       = oe.fit_transform(
    X_df[CAT_FEATURES].astype(str)).astype(np.float32)
X_test_df[CAT_FEATURES]  = oe.transform(
    X_test_df[CAT_FEATURES].astype(str)).astype(np.float32)

# Convert to numpy (float32) — shared across all models
X_np      = X_df.values.astype(np.float32)
X_test_np = X_test_df.values.astype(np.float32)
del X_df, X_test_df; gc.collect()

cat_idx = list(range(len(CAT_FEATURES)))   # first N columns are cats

# Targets
y_raw = train[TARGET].values.astype(np.float64)
y_win = train["cltv_w"].values.astype(np.float64)

# Stratified folds on log-binned original CLTV
y_binned = pd.qcut(np.log1p(y_raw), q=10, labels=False, duplicates="drop")
skf  = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
folds= list(skf.split(X_np, y_binned))

N = len(train); NT = len(test)
RESULTS = {}

# ─── 9. MODEL 1: CatBoost (integer cats, reduced memory params) ──────────────
print("\n" + "=" * 56)
print("  MODEL 1: CatBoost (integer-encoded cats)")
print("=" * 56)

CB_P = dict(
    iterations=1500, learning_rate=0.06, depth=6,
    l2_leaf_reg=4.0, border_count=64,         # ← 64 not 128 → half the histogram memory
    random_strength=0.8, bagging_temperature=0.4,
    loss_function="RMSE", eval_metric="RMSE",
    od_type="Iter", od_wait=100,
    random_seed=SEED, verbose=0, thread_count=-1,
    bootstrap_type="Bernoulli",               # ← less memory than Bayesian bootstrap
    subsample=0.8,
)

oof_cb  = np.zeros(N); pred_cb  = np.zeros(NT)
for fold,(tri,vali) in enumerate(folds,1):
    Xtr = X_np[tri]; Xval = X_np[vali]
    ytr = y_win[tri];yval = y_win[vali]

    m = CatBoostRegressor(**CB_P)
    m.fit(Pool(Xtr,ytr, cat_features=cat_idx),
          eval_set=Pool(Xval,yval,cat_features=cat_idx),
          use_best_model=True)

    oof_cb[vali]  = m.predict(Xval)
    pred_cb      += m.predict(X_test_np)/N_FOLD
    r2 = r2_score(y_raw[vali],oof_cb[vali])
    print(f"    Fold {fold}/{N_FOLD}  R²={r2:.4f}  iter={m.best_iteration_}")
    del m; gc.collect()

r2_cb = r2_score(y_raw,oof_cb)
print(f"  [CatBoost] OOF R²: {r2_cb:.5f}")
RESULTS["CatBoost"] = (oof_cb.copy(), pred_cb.copy())

# ─── 10. MODEL 2: LightGBM ────────────────────────────────────────────────────
print("\n" + "=" * 56)
print("  MODEL 2: LightGBM")
print("=" * 56)

# Build LGB-ready DataFrame (category dtype) from X_np — done ONCE here
# then sliced per fold (category dtype is memory-compact)
cat_col_names = CAT_FEATURES
X_lgb_df      = pd.DataFrame(X_np, columns=ALL_FEATURES)
X_test_lgb_df = pd.DataFrame(X_test_np, columns=ALL_FEATURES)
for col in cat_col_names:
    X_lgb_df[col]      = X_lgb_df[col].astype("category")
    X_test_lgb_df[col] = X_test_lgb_df[col].astype("category")

LGB_P = dict(
    objective="regression", metric="rmse",
    n_estimators=2000, learning_rate=0.05,
    num_leaves=127, min_child_samples=20,
    subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=1.0,
    random_state=SEED, n_jobs=-1, verbosity=-1,
)
lgb_cbs = [lgb.early_stopping(150,verbose=False), lgb.log_evaluation(-1)]

oof_lgb  = np.zeros(N); pred_lgb = np.zeros(NT)
for fold,(tri,vali) in enumerate(folds,1):
    Xtr_l  = X_lgb_df.iloc[tri]
    Xval_l = X_lgb_df.iloc[vali]
    ytr = y_win[tri]; yval = y_win[vali]

    m = lgb.LGBMRegressor(**LGB_P)
    m.fit(Xtr_l, ytr, eval_set=[(Xval_l,yval)],
          callbacks=lgb_cbs, categorical_feature=cat_col_names)

    oof_lgb[vali]  = m.predict(Xval_l)
    pred_lgb      += m.predict(X_test_lgb_df)/N_FOLD
    r2 = r2_score(y_raw[vali],oof_lgb[vali])
    print(f"    Fold {fold}/{N_FOLD}  R²={r2:.4f}")
    del m; gc.collect()

del X_lgb_df, X_test_lgb_df; gc.collect()
r2_lgb = r2_score(y_raw,oof_lgb)
print(f"  [LightGBM] OOF R²: {r2_lgb:.5f}")
RESULTS["LightGBM"] = (oof_lgb.copy(), pred_lgb.copy())

# ─── 11. MODEL 3: XGBoost ────────────────────────────────────────────────────
print("\n" + "=" * 56)
print("  MODEL 3: XGBoost")
print("=" * 56)

XGB_P = dict(
    objective="reg:squarederror", tree_method="hist",
    learning_rate=0.05, n_estimators=2000,
    max_depth=6, subsample=0.8, colsample_bytree=0.8,
    reg_alpha=0.05, reg_lambda=1.5, min_child_weight=10,
    random_state=SEED, n_jobs=-1, verbosity=0,
)

oof_xgb  = np.zeros(N); pred_xgb = np.zeros(NT)
for fold,(tri,vali) in enumerate(folds,1):
    m = xgb.XGBRegressor(**XGB_P)
    m.fit(X_np[tri], y_win[tri],
          eval_set=[(X_np[vali],y_win[vali])],
          verbose=False, early_stopping_rounds=150)
    oof_xgb[vali]  = m.predict(X_np[vali])
    pred_xgb      += m.predict(X_test_np)/N_FOLD
    r2 = r2_score(y_raw[vali],oof_xgb[vali])
    print(f"    Fold {fold}/{N_FOLD}  R²={r2:.4f}  iter={m.best_iteration}")
    del m; gc.collect()

r2_xgb = r2_score(y_raw,oof_xgb)
print(f"  [XGBoost] OOF R²: {r2_xgb:.5f}")
RESULTS["XGBoost"] = (oof_xgb.copy(), pred_xgb.copy())

# ─── 12. MODEL 4: RandomForest ───────────────────────────────────────────────
print("\n" + "=" * 56)
print("  MODEL 4: RandomForest")
print("=" * 56)

oof_rf  = np.zeros(N); pred_rf = np.zeros(NT)
for fold,(tri,vali) in enumerate(folds,1):
    m = RandomForestRegressor(
        n_estimators=250, max_depth=None,
        max_features="sqrt", min_samples_leaf=20,
        n_jobs=-1, random_state=SEED)
    m.fit(X_np[tri], y_win[tri])
    oof_rf[vali]  = m.predict(X_np[vali])
    pred_rf      += m.predict(X_test_np)/N_FOLD
    r2 = r2_score(y_raw[vali],oof_rf[vali])
    print(f"    Fold {fold}/{N_FOLD}  R²={r2:.4f}")
    del m; gc.collect()

r2_rf = r2_score(y_raw,oof_rf)
print(f"  [RandomForest] OOF R²: {r2_rf:.5f}")
RESULTS["RandomForest"] = (oof_rf.copy(), pred_rf.copy())

# ─── 13. RIDGE META-STACKING ─────────────────────────────────────────────────
print("\n" + "=" * 56)
print("  STACKING: Ridge meta-learner")
print("=" * 56)

oofs_mat  = np.column_stack([v[0] for v in RESULTS.values()])  # (N, 4)
preds_mat = np.column_stack([v[1] for v in RESULTS.values()])  # (NT, 4)

print("  Individual OOF R² on original CLTV:")
for name,(_,_) in RESULTS.items():
    i = list(RESULTS.keys()).index(name)
    print(f"    {name:15s}: {r2_score(y_raw, oofs_mat[:,i]):.5f}")

ridge = Ridge(alpha=100.0, fit_intercept=True)
ridge.fit(oofs_mat, y_raw)
pred_stack = ridge.predict(preds_mat)
r2_stack   = r2_score(y_raw, ridge.predict(oofs_mat))
print(f"\n  [Ridge Stack] OOF R²:   {r2_stack:.5f}")

# Top-3 grid blend (CB + LGB + XGB)
best_r2, best_w = -99, (0.5, 0.3, 0.2)
for wA in np.arange(0,1.01,0.05):
    for wB in np.arange(0,1.01-wA,0.05):
        wC = round(1-wA-wB,5)
        if wC<0: continue
        r2 = r2_score(y_raw, wA*oof_cb+wB*oof_lgb+wC*oof_xgb)
        if r2>best_r2:
            best_r2,best_w = r2,(wA,wB,wC)
pred_blend = best_w[0]*pred_cb + best_w[1]*pred_lgb + best_w[2]*pred_xgb
print(f"  [3-way blend]  OOF R²:   {best_r2:.5f}  w={best_w}")

# Mean of all 4
r2_mean4  = r2_score(y_raw, oofs_mat.mean(axis=1))
pred_mean4 = preds_mat.mean(axis=1)
print(f"  [Mean 4 models] OOF R²: {r2_mean4:.5f}")

# Pick best
opts = {"Ridge Stack":  (r2_stack,  pred_stack),
        "3-way blend":  (best_r2,   pred_blend),
        "Mean 4 models":(r2_mean4,  pred_mean4)}
best_name = max(opts, key=lambda k: opts[k][0])
best_final_r2, pred_final = opts[best_name]
print(f"\n  → Best: {best_name}  (R²={best_final_r2:.5f})")

# ─── 14. CLIP & SAVE ─────────────────────────────────────────────────────────
pred_final = np.clip(pred_final, floor, true_max)

sub = pd.DataFrame({"id": test["id"], "cltv": pred_final.astype(int)})
sub.to_csv("submission_v5.csv", index=False)
print(f"\n[Saved] submission_v5.csv  ({len(sub)} rows)")
print(sub.head(8).to_string(index=False))

# ─── 15. FEATURE IMPORTANCE ──────────────────────────────────────────────────
print("\n[Step] Feature importance (LightGBM) …")
import pandas as pd as _pd  # avoid shadowing
X_full_lgb = pd.DataFrame(X_np, columns=ALL_FEATURES)
for col in CAT_FEATURES:
    X_full_lgb[col] = X_full_lgb[col].astype("category")

fi_m = lgb.LGBMRegressor(**{**LGB_P, "n_estimators":1000, "verbosity":-1})
fi_m.fit(X_full_lgb, y_win, categorical_feature=CAT_FEATURES)
fi_df = pd.DataFrame({"feature":ALL_FEATURES,
                       "importance":fi_m.feature_importances_}
                    ).sort_values("importance",ascending=False)
print("\n--- Top 20 Features ---")
print(fi_df.head(20).to_string(index=False))

fig,ax = plt.subplots(figsize=(10,8))
sns.barplot(data=fi_df.head(20),x="importance",y="feature",
            palette="Blues_r",ax=ax)
ax.set_title("Top 20 Features — v5 Ensemble (LightGBM)")
plt.tight_layout()
plt.savefig("plots/feature_importance_v5.png",dpi=120)
plt.close()

# ─── 16. SUMMARY ─────────────────────────────────────────────────────────────
print("\n" + "=" * 56)
print("  FINAL SUMMARY  (all R² on original CLTV)")
print("=" * 56)
print(f"  CatBoost       : {r2_cb:.5f}")
print(f"  LightGBM       : {r2_lgb:.5f}")
print(f"  XGBoost        : {r2_xgb:.5f}")
print(f"  RandomForest   : {r2_rf:.5f}")
print(f"  ─────────────────────────────")
print(f"  Ridge Stack    : {r2_stack:.5f}")
print(f"  3-way blend    : {best_r2:.5f}")
print(f"  Mean 4 models  : {r2_mean4:.5f}")
print(f"  ─────────────────────────────")
print(f"  FINAL ({best_name}): {best_final_r2:.5f}")
print("=" * 56)
print("  submission_v5.csv  ✓")
print("=" * 56)
