"""
VahanBima CLTV — Simple LightGBM + Optuna Hyperparameter Tuning
================================================================
Single model, clean features, Optuna tunes hyperparameters via CV.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import os, gc, time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics         import r2_score
import lightgbm as lgb
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

SEED   = 42
N_FOLD = 5
np.random.seed(SEED)
os.makedirs("plots", exist_ok=True)

# ── 1. LOAD ───────────────────────────────────────────────────────────────────
def _load(name):
    for p in [name, os.path.join("Data", name)]:
        if os.path.exists(p): return pd.read_csv(p)
    raise FileNotFoundError(name)

train = _load("train_data.csv")
test  = _load("test_data.csv")
print(f"Train {train.shape}  Test {test.shape}")
print(f"CLTV: mean={train.cltv.mean():.0f}  std={train.cltv.std():.0f}  max={train.cltv.max():.0f}")

# ── 2. TARGET ─────────────────────────────────────────────────────────────────
cap          = np.percentile(train.cltv, 99)
cltv_min     = train.cltv.min()
cltv_max     = train.cltv.max()
train["y"]   = train.cltv.clip(upper=cap)
print(f"Winsorize cap (99th pct): {cap:.0f}")

# ── 3. FEATURES ───────────────────────────────────────────────────────────────
INCOME = {"<=2L": 1, "2L-5L": 2, "5L-10L": 3, "More than 10L": 4}

def featurize(df):
    d = df.copy()
    d["income_num"] = d["income"].map(INCOME).astype(float)
    d["is_multi"]   = (d["num_policies"] != "1").astype(float)
    d["log_claim"]  = np.log1p(d["claim_amount"])
    d["claim_p_yr"] = d["claim_amount"] / (d["vintage"] + 1)
    d["multi_claim"]= d["is_multi"] * d["claim_amount"]
    d["multi_lclm"] = d["is_multi"] * d["log_claim"]
    d["multi_vint"] = d["is_multi"] * d["vintage"]
    d["multi_inc"]  = d["is_multi"] * d["income_num"]
    for c in ["gender","area","qualification","income","policy","type_of_policy","num_policies"]:
        d[c] = d[c].astype("category")
    return d

train = featurize(train)
test  = featurize(test)

CATS = ["gender","area","qualification","income","policy","type_of_policy","num_policies"]
NUMS = ["marital_status","vintage","claim_amount","income_num","is_multi",
        "log_claim","claim_p_yr","multi_claim","multi_lclm","multi_vint","multi_inc"]
FEATS = CATS + NUMS
print(f"Features: {len(FEATS)}")

X      = train[FEATS]
y_raw  = train["cltv"].values.astype(float)
y_win  = train["y"].values.astype(float)
X_test = test[FEATS]

y_bin = pd.qcut(np.log1p(y_raw), q=10, labels=False, duplicates="drop")
skf   = StratifiedKFold(n_splits=N_FOLD, shuffle=True, random_state=SEED)
folds = list(skf.split(X, y_bin))

# ── 4. OPTUNA TUNING ─────────────────────────────────────────────────────────
print("\n── Optuna hyperparameter search (40 trials × 3-fold) ──")

def objective(trial):
    params = dict(
        objective         = "regression",
        metric            = "rmse",
        verbosity         = -1,
        n_jobs            = -1,
        random_state      = SEED,
        n_estimators      = 2000,
        learning_rate     = trial.suggest_float("lr",          0.02,  0.10, log=True),
        num_leaves        = trial.suggest_int  ("num_leaves",  31,    127),
        min_child_samples = trial.suggest_int  ("min_child",   10,    50),
        subsample         = trial.suggest_float("subsample",   0.6,   1.0),
        colsample_bytree  = trial.suggest_float("colsample",   0.6,   1.0),
        reg_alpha         = trial.suggest_float("reg_alpha",   1e-3,  1.0, log=True),
        reg_lambda        = trial.suggest_float("reg_lambda",  1e-3,  5.0, log=True),
        path_smooth       = trial.suggest_float("path_smooth", 0.0,   2.0),
    )
    # 3-fold quick CV (not 5-fold, for speed)
    kf3  = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    oof  = np.zeros(len(X))
    cbs  = [lgb.early_stopping(80, verbose=False), lgb.log_evaluation(-1)]
    for tri, vali in kf3.split(X, y_bin):
        m = lgb.LGBMRegressor(**params)
        m.fit(X.iloc[tri], y_win[tri],
              eval_set=[(X.iloc[vali], y_win[vali])],
              callbacks=cbs,
              categorical_feature=CATS)
        oof[vali] = m.predict(X.iloc[vali])
    return r2_score(y_raw, oof)          # maximise R² on original scale

study = optuna.create_study(direction="maximize",
                             sampler=optuna.samplers.TPESampler(seed=SEED))
study.optimize(objective, n_trials=40, show_progress_bar=True)

best = study.best_params
print(f"\nBest trial R² (3-fold, original): {study.best_value:.5f}")
print("Best params:")
for k, v in best.items():
    print(f"  {k:18s}: {v}")

# ── 5. FINAL 5-FOLD CV WITH BEST PARAMS ──────────────────────────────────────
print("\n── Final 5-fold CV with best params ──")

FINAL_PARAMS = dict(
    objective         = "regression",
    metric            = "rmse",
    verbosity         = -1,
    n_jobs            = -1,
    random_state      = SEED,
    n_estimators      = 3000,           # more trees; early stopping decides actual count
    learning_rate     = best["lr"],
    num_leaves        = best["num_leaves"],
    min_child_samples = best["min_child"],
    subsample         = best["subsample"],
    colsample_bytree  = best["colsample"],
    reg_alpha         = best["reg_alpha"],
    reg_lambda        = best["reg_lambda"],
    path_smooth       = best["path_smooth"],
)

oof_final  = np.zeros(len(X))
pred_final = np.zeros(len(X_test))
cbs_final  = [lgb.early_stopping(150, verbose=False), lgb.log_evaluation(-1)]

for fold, (tri, vali) in enumerate(folds, 1):
    m = lgb.LGBMRegressor(**FINAL_PARAMS)
    m.fit(X.iloc[tri], y_win[tri],
          eval_set=[(X.iloc[vali], y_win[vali])],
          callbacks=cbs_final,
          categorical_feature=CATS)
    oof_final[vali]  = m.predict(X.iloc[vali])
    pred_final      += m.predict(X_test) / N_FOLD
    r2 = r2_score(y_raw[vali], oof_final[vali])
    print(f"  Fold {fold}/{N_FOLD}  R²={r2:.4f}  iters={m.best_iteration_}")
    del m; gc.collect()

final_r2 = r2_score(y_raw, oof_final)
print(f"\nOOF R² on original CLTV: {final_r2:.5f}")

# ── 6. FEATURE IMPORTANCE ────────────────────────────────────────────────────
m_fi = lgb.LGBMRegressor(**{**FINAL_PARAMS, "n_estimators": 1000})
m_fi.fit(X, y_win, categorical_feature=CATS)
fi = (pd.DataFrame({"feature": FEATS, "importance": m_fi.feature_importances_})
       .sort_values("importance", ascending=False))
print("\nFeature importances:")
print(fi.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(fi["feature"][::-1], fi["importance"][::-1], color="steelblue")
ax.set_title("Feature Importance — LightGBM v7")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig("plots/feature_importance_v7.png", dpi=120)
plt.close()

# ── 7. SUBMISSION ────────────────────────────────────────────────────────────
pred_final = np.clip(pred_final, cltv_min, cltv_max)
sub = pd.DataFrame({"id": test["id"], "cltv": pred_final.astype(int)})
sub.to_csv("submission_v7.csv", index=False)
print(f"\nSaved submission_v7.csv  ({len(sub)} rows)")
print(f"Final OOF R²: {final_r2:.5f}")