"""SHAP figures for the tuned XGBoost model, explained on the 2023-2024 test fires.

Rebuilds the model from the settings the randomized search picked in notebooks/02_models.ipynb,
checks it reproduces the notebook's test scores, then saves three figures to figures/.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from common import PROCESSED, ROOT

D = PROCESSED / "model_ready"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

X_train = pd.read_csv(D / "X_train.csv")
y_train = pd.read_csv(D / "y_train.csv").iloc[:, 0]
X_test = pd.read_csv(D / "X_test.csv")
y_test = pd.read_csv(D / "y_test.csv").iloc[:, 0]
hard = pd.read_csv(D / "info_test.csv")["hard_case"].values == 1

model = XGBClassifier(
    colsample_bytree=0.5262295366443046, learning_rate=0.004714768498224131, max_depth=3,
    min_child_weight=4, n_estimators=1108, reg_lambda=2.05135050271414, subsample=0.7754406053849281,
    random_state=0, n_jobs=4,
).fit(X_train, y_train)

p = model.predict_proba(X_test)[:, 1]
print(f"test PR-AUC {average_precision_score(y_test, p):.3f}, "
      f"hard {average_precision_score(y_test[hard], p[hard]):.3f}  (notebook: 0.554, 0.454)")

shap_values = shap.TreeExplainer(model)(X_test)

shap.plots.bar(shap_values, max_display=15, show=False)
plt.title("XGBoost: average impact of each input (2023–2024 test fires)", loc="left", fontsize=11)
plt.savefig(FIG / "shap_bar.png", bbox_inches="tight", dpi=150)
plt.close("all")

shap.plots.beeswarm(shap_values, max_display=15, show=False)
plt.title("XGBoost: how each input pushes each fire (right = toward a big fire)", loc="left", fontsize=11)
plt.savefig(FIG / "shap_beeswarm.png", bbox_inches="tight", dpi=150)
plt.close("all")


panels = [("fp_n_pixels", "hot pixels on first pass", True), ("elevation_m", "elevation (m)", False),
          ("slope_deg", "slope (degrees)", False), ("rhmin_prev", "minimum humidity, day before (%)", False),
          ("vpd_prev", "air dryness (VPD), day before (kPa)", False), ("local_hour", "local hour of first detection", False)]
fig, axes = plt.subplots(2, 3, figsize=(15, 8.5))
for ax, (col, label, unlog) in zip(axes.ravel(), panels):
    x = X_test[col].values
    x = np.expm1(x) if unlog else x
    ax.scatter(x, shap_values[:, col].values, s=8, alpha=0.5, color="#c2410c", linewidths=0)
    ax.axhline(0, color="#1d2620", lw=0.8, ls="--")
    if unlog:
        ax.set_xscale("log")
    ax.set_xlabel(label)
    ax.set_ylabel("SHAP value (→ big fire)")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
fig.suptitle("XGBoost dependence plots: one dot per test fire. Above 0 = pushed toward a big fire",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG / "shap_dependence.png", dpi=150)
print("saved", sorted(f.name for f in FIG.glob("shap_*.png")))
