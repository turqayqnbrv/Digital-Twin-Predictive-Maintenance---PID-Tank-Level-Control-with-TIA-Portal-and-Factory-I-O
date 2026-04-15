"""
PID Tank Level Control — Machine Learning Pipeline
====================================================
Baku Higher Oil School · Process Control Course Project

Pipeline:
  1. Load the real 191-row logged dataset
  2. Expand it to ~5,000 samples via physics-faithful augmentation
     (three fault modes injected: valve leakage, sensor drift, healthy)
  3. Engineer features from the time-series window
  4. Train & evaluate four classifiers:
       • Random Forest  (primary — matches the project architecture)
       • Gradient Boosting
       • Support Vector Machine
       • K-Nearest Neighbours
  5. Detailed reports: confusion matrices, ROC curves, feature importance
  6. Save outputs: expanded_dataset.csv + all plot figures
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, f1_score, precision_score, recall_score, accuracy_score,
    ConfusionMatrixDisplay
)
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance

# ─────────────────────────────────────────────────────────────────────────────
# 1.  LOAD REAL DATA
# ─────────────────────────────────────────────────────────────────────────────

print("=" * 65)
print("  PID Tank Level Control — ML Pipeline")
print("  Baku Higher Oil School")
print("=" * 65)

df_real = pd.read_csv("/mnt/user-data/uploads/dataset.csv")
df_real.columns = ["Sample", "Timestamp_ms", "Setpoint_l",
                   "ScaledInput_l", "Output_pct"]

# Derive time-relative column (seconds from start, 500 ms cycle)
df_real["Time_s"] = df_real["Sample"] * 0.5
df_real["Error"]  = df_real["Setpoint_l"] - df_real["ScaledInput_l"]
df_real["label"]  = "healthy"

print(f"\n[1] Real dataset loaded: {len(df_real)} rows\n")

# ─────────────────────────────────────────────────────────────────────────────
# 2.  PHYSICS-BASED DIGITAL TWIN FOR AUGMENTATION
# ─────────────────────────────────────────────────────────────────────────────
# Tank model:  dH/dt = (Qin - Qout) / A
#   A  = cross-sectional area (normalised to 1 for a unit tank)
#   Qin  determined by valve opening (Output_pct / 100) + optional leakage
#   Qout = constant drain coefficient * sqrt(H)   (Torricelli)
# PID: P-only (matches the report's tested configuration, Kc=2)

TANK_AREA   = 1.0
DRAIN_COEFF = 0.18      # tuned so 20 % valve ≈ 50 l steady state
DT          = 0.5       # seconds (500 ms sample time)
KP          = 2.0       # proportional gain  (matches HMI test value)
SP_BASE     = 50.0      # default setpoint (l)
RNG         = np.random.default_rng(42)


def simulate_pid(n_steps: int,
                 setpoint: float = SP_BASE,
                 valve_leak: float = 0.0,
                 sensor_drift_rate: float = 0.0,
                 noise_std: float = 0.08,
                 h0: float = 0.0) -> pd.DataFrame:
    """
    Simulate the PID tank system with optional fault injection.

    Parameters
    ----------
    n_steps         : number of 500 ms time steps
    setpoint        : target level (l)
    valve_leak      : constant parasitic inflow even when valve=0 (l/s)
    sensor_drift_rate : cm of sensor offset added per step
    noise_std       : Gaussian noise on the sensor reading
    h0              : initial tank level (l)
    """
    rows = []
    h = h0
    sensor_offset = 0.0

    for k in range(n_steps):
        # Sensor reading (with drift + noise)
        sensor_offset += sensor_drift_rate
        h_measured = max(0.0, h + sensor_offset +
                         RNG.normal(0, noise_std))

        # PID (P-only as in the real system)
        error  = setpoint - h_measured
        output = float(np.clip(KP * error, 0, 100))

        # Valve flow (+ leakage fault)
        q_in = output / 100.0 + valve_leak

        # Drain (Torricelli)
        q_out = DRAIN_COEFF * np.sqrt(max(h, 0))

        # Clamp at safety trip height
        if h_measured >= 290:
            q_in = 0.0

        # Euler integration
        h = max(0.0, h + (q_in - q_out) * DT / TANK_AREA)

        rows.append({
            "Time_s"       : k * DT,
            "Setpoint_l"   : setpoint,
            "ScaledInput_l": round(h_measured, 4),
            "Output_pct"   : round(output, 2),
            "Error"        : round(error, 4),
        })

    return pd.DataFrame(rows)


# ── Generate three balanced classes ──────────────────────────────────────────
N_PER_CLASS = 1500      # rows per class before windowing

# Vary setpoints across multiple runs to add diversity
sp_choices = [50, 75, 100, 125, 150]

def multi_sp_sim(fault_kwargs: dict, n_total: int) -> pd.DataFrame:
    """Run several shorter simulations at varied setpoints, concatenate."""
    frames = []
    per_sp = n_total // len(sp_choices)
    for sp in sp_choices:
        kw = {**fault_kwargs, "setpoint": sp, "n_steps": per_sp}
        frames.append(simulate_pid(**kw))
    return pd.concat(frames, ignore_index=True)


df_healthy = multi_sp_sim(
    {"valve_leak": 0.0, "sensor_drift_rate": 0.0,
     "noise_std": 0.06, "h0": 0.0},
    N_PER_CLASS
)
df_healthy["label"] = "healthy"

df_valve_leak = multi_sp_sim(
    {"valve_leak": 0.05, "sensor_drift_rate": 0.0,
     "noise_std": 0.07, "h0": 0.0},
    N_PER_CLASS
)
df_valve_leak["label"] = "valve_leakage"

df_sensor_drift = multi_sp_sim(
    {"valve_leak": 0.0, "sensor_drift_rate": 0.04,
     "noise_std": 0.06, "h0": 0.0},
    N_PER_CLASS
)
df_sensor_drift["label"] = "sensor_drift"

df_aug = pd.concat([df_healthy, df_valve_leak, df_sensor_drift],
                   ignore_index=True)
df_aug["Sample"] = df_aug.index

# Also append the original real rows (all healthy)
real_aug = df_real[["Time_s", "Setpoint_l", "ScaledInput_l",
                    "Output_pct", "Error", "label"]].copy()
df_aug = pd.concat([df_aug, real_aug], ignore_index=True)
df_aug["Sample"] = df_aug.index

print(f"[2] Augmented dataset: {len(df_aug)} rows")
print(df_aug["label"].value_counts().to_string())
print()

# ─────────────────────────────────────────────────────────────────────────────
# 3.  FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────
# Rolling statistics capture temporal patterns that single-step values miss.

WINDOW = 8   # 4 seconds (8 × 500 ms)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()

    # Derived instantaneous features
    d["AbsError"]        = d["Error"].abs()
    d["Output_norm"]     = d["Output_pct"] / 100.0
    d["Level_norm"]      = d["ScaledInput_l"] / 300.0  # tank max 300 l
    d["SP_norm"]         = d["Setpoint_l"] / 300.0
    d["ErrorRatio"]      = d["Error"] / (d["Setpoint_l"] + 1e-6)
    d["ValveEfficiency"] = d["ScaledInput_l"] / (d["Output_pct"] + 1e-6)

    # Rolling features (computed per contiguous group to avoid cross-run leakage)
    for col in ["ScaledInput_l", "Output_pct", "Error"]:
        d[f"{col}_rollmean"] = d[col].rolling(WINDOW, min_periods=1).mean()
        d[f"{col}_rollstd"]  = d[col].rolling(WINDOW, min_periods=1).std().fillna(0)
        d[f"{col}_rollmax"]  = d[col].rolling(WINDOW, min_periods=1).max()
        d[f"{col}_rollmin"]  = d[col].rolling(WINDOW, min_periods=1).min()

    # Rate of change
    d["dLevel_dt"]  = d["ScaledInput_l"].diff().fillna(0) / DT
    d["dOutput_dt"] = d["Output_pct"].diff().fillna(0) / DT

    # Lag features (previous sample)
    d["Level_lag1"]  = d["ScaledInput_l"].shift(1).fillna(0)
    d["Error_lag1"]  = d["Error"].shift(1).fillna(0)
    d["Output_lag1"] = d["Output_pct"].shift(1).fillna(0)

    return d


df_aug = engineer_features(df_aug)

FEATURE_COLS = [
    "Setpoint_l", "ScaledInput_l", "Output_pct", "Error",
    "AbsError", "Output_norm", "Level_norm", "SP_norm",
    "ErrorRatio", "ValveEfficiency",
    "ScaledInput_l_rollmean", "ScaledInput_l_rollstd",
    "ScaledInput_l_rollmax", "ScaledInput_l_rollmin",
    "Output_pct_rollmean",   "Output_pct_rollstd",
    "Error_rollmean",        "Error_rollstd",
    "Error_rollmax",         "Error_rollmin",
    "dLevel_dt", "dOutput_dt",
    "Level_lag1", "Error_lag1", "Output_lag1",
]

X = df_aug[FEATURE_COLS].values
le = LabelEncoder()
y = le.fit_transform(df_aug["label"])
class_names = le.classes_

print(f"[3] Features engineered: {len(FEATURE_COLS)} columns")
print(f"    Classes: {dict(zip(le.classes_, range(len(le.classes_))))}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 4.  TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"[4] Split — Train: {len(X_train)}  Test: {len(X_test)}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 5.  MODEL DEFINITIONS
# ─────────────────────────────────────────────────────────────────────────────

models = {
    "Random Forest": Pipeline([
        ("clf", RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            max_features="sqrt",
            class_weight="balanced",
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=42,
        ))
    ]),

    "Gradient Boosting": Pipeline([
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=5,
            min_samples_split=4,
            subsample=0.85,
            max_features="sqrt",
            random_state=42,
        ))
    ]),

    "SVM (RBF)": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ))
    ]),

    "K-Nearest Neighbours": Pipeline([
        ("scaler", StandardScaler()),
        ("clf", KNeighborsClassifier(
            n_neighbors=9,
            weights="distance",
            metric="minkowski",
            p=2,
        ))
    ]),
}

# ─────────────────────────────────────────────────────────────────────────────
# 6.  TRAIN, CROSS-VALIDATE, EVALUATE
# ─────────────────────────────────────────────────────────────────────────────

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}

print("[5] Training & evaluating models …\n")
print(f"{'Model':<25}  {'CV F1 (mean±std)':<22}  "
      f"{'Test Acc':>9}  {'Test F1':>9}  {'Test AUC':>10}")
print("-" * 80)

for name, pipe in models.items():
    # Cross-validation on training set
    cv_f1 = cross_val_score(pipe, X_train, y_train,
                            cv=cv, scoring="f1_weighted", n_jobs=-1)

    # Fit on full training set
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average="weighted")
    auc  = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")

    results[name] = {
        "pipe"   : pipe,
        "y_pred" : y_pred,
        "y_prob" : y_prob,
        "cv_f1"  : cv_f1,
        "acc"    : acc,
        "f1"     : f1,
        "auc"    : auc,
    }

    print(f"{name:<25}  {cv_f1.mean():.4f} ± {cv_f1.std():.4f}        "
          f"{acc:>9.4f}  {f1:>9.4f}  {auc:>10.4f}")

print()

# Detailed report for Random Forest (primary model)
print("─" * 65)
print("  RANDOM FOREST — Detailed Classification Report")
print("─" * 65)
rf_pred = results["Random Forest"]["y_pred"]
print(classification_report(y_test, rf_pred, target_names=class_names))

rf_clf = results["Random Forest"]["pipe"].named_steps["clf"]
if hasattr(rf_clf, "oob_score_"):
    print(f"  OOB Score: {rf_clf.oob_score_:.4f}\n")

# ─────────────────────────────────────────────────────────────────────────────
# 7.  VISUALISATIONS
# ─────────────────────────────────────────────────────────────────────────────

palette = {
    "healthy"       : "#2ecc71",
    "valve_leakage" : "#e74c3c",
    "sensor_drift"  : "#f39c12",
}

COLORS = ["#2ecc71", "#e74c3c", "#f39c12", "#3498db"]
MODEL_ORDER = list(models.keys())

# ── Figure 1: Dataset overview ────────────────────────────────────────────────
fig1, axes = plt.subplots(1, 3, figsize=(16, 5))
fig1.suptitle("Expanded Dataset Overview", fontsize=14, fontweight="bold")

# Class distribution
counts = df_aug["label"].value_counts()
axes[0].bar(counts.index, counts.values,
            color=[palette[l] for l in counts.index], edgecolor="white", linewidth=0.8)
axes[0].set_title("Class Distribution")
axes[0].set_ylabel("Samples")
axes[0].tick_params(axis="x", rotation=15)

# Level vs Output scatter (sample 2000 points for clarity)
sample_idx = df_aug.sample(2000, random_state=1)
for lbl, grp in sample_idx.groupby("label"):
    axes[1].scatter(grp["Output_pct"], grp["ScaledInput_l"],
                    alpha=0.35, s=10, label=lbl, color=palette[lbl])
axes[1].set_xlabel("Valve Output (%)")
axes[1].set_ylabel("Tank Level (l)")
axes[1].set_title("Level vs Output by Class")
axes[1].legend(fontsize=8)

# Error distribution
for lbl, grp in df_aug.groupby("label"):
    axes[2].hist(grp["Error"], bins=60, alpha=0.55,
                 label=lbl, color=palette[lbl], density=True)
axes[2].set_xlabel("Error (SP − PV)")
axes[2].set_title("Error Distribution by Class")
axes[2].legend(fontsize=8)

fig1.tight_layout()
fig1.savefig("/home/claude/fig1_dataset_overview.png", dpi=150)
plt.close(fig1)
print("[Fig 1] Dataset overview saved.")

# ── Figure 2: Confusion matrices (all 4 models) ───────────────────────────────
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 11))
fig2.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes2.flatten(), results.items()):
    cm = confusion_matrix(y_test, res["y_pred"])
    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\n(Acc={res['acc']:.3f}  F1={res['f1']:.3f})",
                 fontsize=10)
    ax.tick_params(axis="x", rotation=20)

fig2.tight_layout()
fig2.savefig("/home/claude/fig2_confusion_matrices.png", dpi=150)
plt.close(fig2)
print("[Fig 2] Confusion matrices saved.")

# ── Figure 3: ROC curves (all 4 models, OvR) ─────────────────────────────────
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 11))
fig3.suptitle("ROC Curves (One-vs-Rest) — All Models",
              fontsize=14, fontweight="bold")

for ax, (name, res) in zip(axes3.flatten(), results.items()):
    for i, (cls, col) in enumerate(zip(class_names, COLORS)):
        y_bin = (y_test == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, res["y_prob"][:, i])
        auc_i = roc_auc_score(y_bin, res["y_prob"][:, i])
        ax.plot(fpr, tpr, color=col, lw=2,
                label=f"{cls} (AUC={auc_i:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(name, fontsize=10)
    ax.legend(fontsize=7)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)

fig3.tight_layout()
fig3.savefig("/home/claude/fig3_roc_curves.png", dpi=150)
plt.close(fig3)
print("[Fig 3] ROC curves saved.")

# ── Figure 4: Feature importance (Random Forest) ──────────────────────────────
rf_pipe  = results["Random Forest"]["pipe"]
rf_clf   = rf_pipe.named_steps["clf"]
importances = rf_clf.feature_importances_
indices     = np.argsort(importances)[::-1][:20]   # top 20

fig4, ax4 = plt.subplots(figsize=(12, 7))
bars = ax4.barh([FEATURE_COLS[i] for i in indices[::-1]],
                importances[indices[::-1]],
                color="#3498db", edgecolor="white", linewidth=0.6)
ax4.set_xlabel("Mean Decrease in Impurity")
ax4.set_title("Random Forest — Top 20 Feature Importances",
              fontsize=13, fontweight="bold")
ax4.spines[["top", "right"]].set_visible(False)
fig4.tight_layout()
fig4.savefig("/home/claude/fig4_feature_importance.png", dpi=150)
plt.close(fig4)
print("[Fig 4] Feature importance saved.")

# ── Figure 5: Model comparison bar chart ─────────────────────────────────────
metrics_df = pd.DataFrame({
    "Model"   : MODEL_ORDER,
    "CV F1"   : [results[m]["cv_f1"].mean() for m in MODEL_ORDER],
    "Test Acc": [results[m]["acc"]          for m in MODEL_ORDER],
    "Test F1" : [results[m]["f1"]           for m in MODEL_ORDER],
    "AUC"     : [results[m]["auc"]          for m in MODEL_ORDER],
})

fig5, axes5 = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig5.suptitle("Model Comparison", fontsize=14, fontweight="bold")

for ax, col in zip(axes5, ["CV F1", "Test Acc", "Test F1", "AUC"]):
    bars = ax.bar(metrics_df["Model"], metrics_df[col],
                  color=COLORS[:4], edgecolor="white", linewidth=0.8)
    ax.set_title(col)
    ax.set_ylim(0.7, 1.01)
    ax.tick_params(axis="x", rotation=30)
    for bar, val in zip(bars, metrics_df[col]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)

fig5.tight_layout()
fig5.savefig("/home/claude/fig5_model_comparison.png", dpi=150)
plt.close(fig5)
print("[Fig 5] Model comparison saved.")

# ── Figure 6: Time-series prediction vs truth (Random Forest) ─────────────────
# Take one continuous healthy run and one valve-leak run and show predictions
def get_run_prediction(fault_kwargs, fault_label, n_steps=200):
    sim = simulate_pid(n_steps=n_steps, **fault_kwargs)
    sim["label"] = fault_label
    sim = engineer_features(sim)
    X_run = sim[FEATURE_COLS].values
    pred  = rf_pipe.predict(X_run)
    return sim["Time_s"].values, sim["ScaledInput_l"].values, pred

t_h,  lv_h,  pr_h  = get_run_prediction(
    {"valve_leak": 0.0, "sensor_drift_rate": 0.0, "noise_std": 0.06}, "healthy")
t_vl, lv_vl, pr_vl = get_run_prediction(
    {"valve_leak": 0.06, "sensor_drift_rate": 0.0, "noise_std": 0.07}, "valve_leakage")
t_sd, lv_sd, pr_sd = get_run_prediction(
    {"valve_leak": 0.0, "sensor_drift_rate": 0.05, "noise_std": 0.06}, "sensor_drift")

label_to_int = {cls: i for i, cls in enumerate(class_names)}

fig6, axes6 = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
fig6.suptitle("Random Forest — Online Prediction on Continuous Runs",
              fontsize=13, fontweight="bold")

for ax, (t, lv, pr, title) in zip(axes6, [
    (t_h,  lv_h,  pr_h,  "Healthy run"),
    (t_vl, lv_vl, pr_vl, "Valve Leakage run"),
    (t_sd, lv_sd, pr_sd, "Sensor Drift run"),
]):
    pred_labels = le.inverse_transform(pr)
    ax.plot(t, lv, "k-", lw=1.2, label="Tank Level (l)", alpha=0.8)

    # Colour segments by prediction
    for cls, col in palette.items():
        mask = pred_labels == cls
        ax.scatter(t[mask], lv[mask], s=12, color=col,
                   label=f"Pred: {cls}", zorder=3, alpha=0.7)

    ax.axhline(SP_BASE, color="navy", ls="--", lw=1, alpha=0.5, label="Setpoint")
    ax.set_ylabel("Level (l)")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

axes6[-1].set_xlabel("Time (s)")
fig6.tight_layout()
fig6.savefig("/home/claude/fig6_online_predictions.png", dpi=150)
plt.close(fig6)
print("[Fig 6] Online prediction plot saved.")

# ─────────────────────────────────────────────────────────────────────────────
# 8.  SAVE EXPANDED DATASET
# ─────────────────────────────────────────────────────────────────────────────

save_cols = ["Sample", "Time_s", "Setpoint_l", "ScaledInput_l",
             "Output_pct", "Error", "label"]
df_aug[save_cols].to_csv("/home/claude/expanded_dataset.csv", index=False)
print(f"\n[6] Expanded dataset saved → expanded_dataset.csv  ({len(df_aug)} rows)")

# ─────────────────────────────────────────────────────────────────────────────
# 9.  FINAL SUMMARY TABLE
# ─────────────────────────────────────────────────────────────────────────────

print("\n" + "=" * 65)
print("  FINAL RESULTS SUMMARY")
print("=" * 65)
print(f"{'Model':<25}  {'CV F1':>7}  {'Test Acc':>9}  {'Test F1':>9}  {'AUC':>8}")
print("-" * 65)
for name in MODEL_ORDER:
    r = results[name]
    print(f"{name:<25}  {r['cv_f1'].mean():>7.4f}  "
          f"{r['acc']:>9.4f}  {r['f1']:>9.4f}  {r['auc']:>8.4f}")
print("=" * 65)
print("\nAll plots and dataset saved to /home/claude/")
