# ============================================================
# Bar Crawl TAC Analysis -- Four-Model OLS Comparison + Plots
# Replicates the models, tables, and figures reported in
# Sections 3 and 4 (Figures 1-3, Table 2).
# ============================================================
 
import pandas as pd, numpy as np, statsmodels.api as sm
import matplotlib.pyplot as plt
from scipy import stats as st
 
# ------------------------------------------------------------
# 1. Load the pre-processed, window-level dataset.
#    Each row = one 10-second accelerometer window matched to
#    the nearest TAC (blood alcohol) reading.
# ------------------------------------------------------------
data = pd.read_csv("bcdhd/analysis_data.csv")
 
# Convert the categorical phone-type column into a 0/1 numeric
# indicator (1 = Android, 0 = iPhone) so it can be used as a
# regression predictor.
data["phonetype_Android"] = (data["phonetype"] == "Android").astype(float)
 
# The response variable (Y) we are trying to predict: TAC.
y = data["TAC_Reading"]
 
# ------------------------------------------------------------
# 2. Helper function: adds an intercept column and fits an
#    Ordinary Least Squares (OLS) linear regression model.
#    Reused for all four model specifications below.
# ------------------------------------------------------------
def fit(Xdf):
    X = sm.add_constant(Xdf)      # adds the intercept (beta_0) term
    return sm.OLS(y, X).fit()     # fits Y = beta_0 + beta_1*X1 + ...
 
# ------------------------------------------------------------
# M1: Simple linear regression -- TAC ~ svm_max
# The baseline model: predicts TAC from peak movement magnitude alone.
# ------------------------------------------------------------
m1 = fit(data[["svm_max"]])
print(m1.summary())
 
# ------------------------------------------------------------
# M2: Multiple linear regression
# TAC ~ svm_max + x_std + y_std + z_std + phonetype
# Adds the three per-axis wobble features and phone type to see
# whether extra movement detail improves the fit.
# ------------------------------------------------------------
m2 = fit(data[["svm_max","x_std","y_std","z_std","phonetype_Android"]])
print(m2.summary())
 
# ------------------------------------------------------------
# M3: Interaction model -- TAC ~ svm_max * phonetype_Android
# Lets the effect of svm_max on TAC differ between Android and
# iPhone users, by adding a multiplicative interaction term.
# ------------------------------------------------------------
d3 = data[["svm_max","phonetype_Android"]].copy()
d3["svm_max:phonetype_Android"] = d3["svm_max"] * d3["phonetype_Android"]
m3 = fit(d3)
print(m3.summary())
 
# ------------------------------------------------------------
# M4: Polynomial model
# TAC ~ svm_max + svm_max^2 + x_std + y_std + z_std + phonetype
# Adds a squared svm_max term to test for curvature (non-linearity)
# in the movement-to-TAC relationship.
# ------------------------------------------------------------
d4 = data[["svm_max","x_std","y_std","z_std","phonetype_Android"]].copy()
d4["I(svm_max^2)"] = data["svm_max"]**2
m4 = fit(d4)
print(m4.summary())
 
# ------------------------------------------------------------
# 3. Example predictions from the simple model (M1), with 95%
#    confidence intervals (uncertainty in the average TAC) and
#    prediction intervals (uncertainty for a single new window)
#    at a few representative svm_max values.
# ------------------------------------------------------------
newX = sm.add_constant(pd.DataFrame({"svm_max":[1, 5, 10]}), has_constant="add")
pred = m1.get_prediction(newX)
print(pred.summary_frame(alpha=0.05))
 
# ============================================================
# 4. FIGURE 1 -- Pairwise correlation / scatterplot matrix
#    Grid of the four movement features + TAC: diagonal shows
#    each variable's distribution, upper triangle shows
#    correlation coefficients, lower triangle shows scatterplots.
# ============================================================
vars_ = ["svm_max", "x_std", "y_std", "z_std", "TAC_Reading"]
labels = ["svm_max", "x_std", "y_std", "z_std", "TAC"]
d = data[vars_]
d_plot = d.sample(n=min(6000, len(d)), random_state=1)  # subsample for speed
 
def stars(p):
    if p < 0.001: return "***"
    if p < 0.01: return "**"
    if p < 0.05: return "*"
    return ""
 
n = len(vars_)
fig, axes = plt.subplots(n, n, figsize=(8.2, 8.2))
fig.suptitle("Accelerometer Feature and TAC: Pairwise Feature Matrix",
             fontsize=11, fontweight="bold", y=0.995)
 
for i in range(n):
    for j in range(n):
        ax = axes[i, j]
        if i == j:
            ax.hist(d[vars_[i]], bins=30, color="#3498db", alpha=0.6, density=True)
            ax.set_yticks([])
        elif i > j:
            ax.scatter(d_plot[vars_[j]], d_plot[vars_[i]], s=2, alpha=0.12, color="#2c3e50")
        else:
            r, p = st.pearsonr(d[vars_[j]], d[vars_[i]])
            ax.text(0.5, 0.6, f"Corr:\n{r:.3f}{stars(p)}", ha="center", va="center",
                    fontsize=9, transform=ax.transAxes)
            ax.set_xticks([]); ax.set_yticks([])
        if i == n - 1:
            ax.set_xlabel(labels[j], fontsize=8)
        if j == 0:
            ax.set_ylabel(labels[i], fontsize=8)
 
plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.savefig("bcdhd/figs/pairwise_matrix.png", dpi=170)
plt.show()   # display Figure 1
plt.close()
 
# ============================================================
# 5. FIGURE 2 -- Simple linear regression plot (M1)
#    Scatter of every window (TAC vs. svm_max) with the fitted
#    OLS line overlaid.
# ============================================================
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(data["svm_max"], data["TAC_Reading"], s=4, alpha=0.15, color="#3B6FA0")
xs = np.linspace(data["svm_max"].min(), data["svm_max"].quantile(0.99), 100)
ax.plot(xs, m1.params["const"] + m1.params["svm_max"] * xs, "r--", linewidth=2.5)
ax.set_xlim(0, data["svm_max"].quantile(0.99))
ax.set_xlabel("svm_max (peak acceleration magnitude in 10s window, g)")
ax.set_ylabel("TAC Reading (g/dl)")
ax.set_title("Simple linear regression: TAC ~ svm_max")
plt.tight_layout()
plt.savefig("bcdhd/figs/simple_fit.png")
plt.show()   # display Figure 2
plt.close()
 
# ============================================================
# 6. FIGURE 3 -- Predicted TAC vs. svm_max across all four models
#    M2/M4 hold the wobble features (x_std, y_std, z_std) at their
#    sample means, phone type = iPhone. M3 is drawn as two lines
#    (iPhone vs. Android) to make its interaction effect visible.
# ============================================================
x_max = data["svm_max"].quantile(0.95)
xs = np.linspace(0, x_max, 200)
x_std_mean, y_std_mean, z_std_mean = data[["x_std","y_std","z_std"]].mean()
 
fig, ax = plt.subplots(figsize=(7.2, 5))
 
pred1 = m1.params["const"] + m1.params["svm_max"] * xs
ax.plot(xs, pred1, label="M1 Simple", color="#1f77b4", linewidth=2.2)
 
pred2 = (m2.params["const"] + m2.params["svm_max"]*xs + m2.params["x_std"]*x_std_mean
         + m2.params["y_std"]*y_std_mean + m2.params["z_std"]*z_std_mean)
ax.plot(xs, pred2, label="M2 Multiple (iPhone, avg wobble)", color="#2ca02c", linewidth=2.2)
 
pred3_iphone = m3.params["const"] + m3.params["svm_max"] * xs
pred3_android = (m3.params["const"] + m3.params["phonetype_Android"]
                  + (m3.params["svm_max"] + m3.params["svm_max:phonetype_Android"]) * xs)
ax.plot(xs, pred3_iphone, label="M3 Interaction (iPhone)", color="#d62728", linewidth=2.2, linestyle="--")
ax.plot(xs, pred3_android, label="M3 Interaction (Android)", color="#d62728", linewidth=2.2, linestyle=":")
 
pred4 = (m4.params["const"] + m4.params["svm_max"]*xs + m4.params["I(svm_max^2)"]*xs**2
         + m4.params["x_std"]*x_std_mean + m4.params["y_std"]*y_std_mean
         + m4.params["z_std"]*z_std_mean)
ax.plot(xs, pred4, label="M4 Polynomial (iPhone, avg wobble)", color="#9467bd", linewidth=2.2)
 
ax.set_xlabel("svm_max (peak acceleration magnitude in 10s window, g)")
ax.set_ylabel("Predicted TAC Reading (g/dl)")
ax.set_title("Predicted TAC vs. svm_max Across All Four Models")
ax.legend(fontsize=8, loc="upper left")
ax.set_xlim(0, x_max)
plt.tight_layout()
plt.savefig("bcdhd/figs/model_comparison_lines.png")
plt.show()   # display Figure 3
plt.close()
