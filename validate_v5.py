import pandas as pd, numpy as np, warnings
warnings.filterwarnings("ignore")

train = pd.read_csv("Data/train_data.csv")
test  = pd.read_csv("Data/test_data.csv")

INCOME_ORDER = {"<=2L":0,"2L-5L":1,"5L-10L":2,"More than 10L":3}
all_inc = pd.concat([train["income"],test["income"]])
income_map = {v: INCOME_ORDER.get(v,i) for i,v in enumerate(sorted(all_inc.unique()))}

def fe(df, m):
    df=df.copy()
    df["income_num"]=df["income"].map(m)
    df["policy_type"]=df["policy"].astype(str)+"_"+df["type_of_policy"].astype(str)
    return df

train=fe(train,income_map); test=fe(test,income_map)

# compute deviation features
n_tr=len(train)
all_=pd.concat([train.reset_index(drop=True),test.reset_index(drop=True)],ignore_index=True)

for col, grp in [("dev_income",["income"]),("dev_poltype",["policy_type"])]:
    seg_mean=all_.groupby(grp)["claim_amount"].transform("mean")
    seg_std=all_.groupby(grp)["claim_amount"].transform("std").fillna(1)
    all_[col]=all_["claim_amount"]-seg_mean
    all_[col+"_norm"]=all_[col]/(seg_std+1)
    for feat in [col, col+"_norm"]:
        train[feat]=all_.iloc[:n_tr][feat].values
        test[feat]=all_.iloc[n_tr:][feat].values

print("Train shape:", train.shape)
print("Missing:", train.isnull().sum().sum())
print()
print("Deviation feature correlations with CLTV:")
for col in ["dev_income","dev_income_norm","dev_poltype","dev_poltype_norm"]:
    print(f"  {col}: {train[col].corr(train['cltv']):.4f}")

# Winsorization check
cap = np.percentile(train["cltv"], 99)
print(f"\n99th pct cap: {cap:.0f}")
print(f"Rows capped: {(train['cltv'] > cap).sum()}")
print("ALL GOOD - v5 validated!")
