"""
AgriTrust AI – Agricultural Loan Credit Scorer
================================================
train_model.py

Generates a synthetic agricultural dataset, trains an XGBoost binary
classifier to predict loan approval, evaluates it, and persists the
trained model to disk.
"""

# ── 1. Imports ────────────────────────────────────────────────────────────────
import os
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# ── 2. Reproducibility seed ───────────────────────────────────────────────────
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# ── 3. Synthetic dataset generation ──────────────────────────────────────────
def generate_dataset(n_samples: int = 600) -> pd.DataFrame:
    """
    Generate a realistic synthetic dataset for agricultural loan scoring.

    Feature logic (domain-inspired rules):
    - Larger farms with better soil & rainfall → higher yields
    - Good payment history (few previous_loans that defaulted) → approval more likely
    - A probabilistic approval label is derived from a weighted score so
      the dataset is not perfectly separable (mimics real-world noise).
    """

    # --- Raw features ---
    farm_size     = np.round(np.random.uniform(0.5, 50.0, n_samples), 2)   # acres
    soil_score    = np.random.randint(20, 101, n_samples)                   # 0–100 index
    rainfall      = np.round(np.random.uniform(200, 1500, n_samples), 1)   # mm/year
    previous_loans = np.random.randint(0, 6, n_samples)                    # count

    # Yield is loosely correlated with farm size, soil quality & rainfall
    yield_amount = np.round(
        0.05 * farm_size
        + 0.03 * soil_score
        + 0.002 * rainfall
        + np.random.normal(0, 0.5, n_samples),
        2,
    )
    yield_amount = np.clip(yield_amount, 0.1, None)  # no negative yields

    # --- Approval label (probabilistic) ---
    # Build a continuous "creditworthiness" score, then add noise
    score = (
        0.30 * (farm_size / 50)
        + 0.25 * (soil_score / 100)
        + 0.20 * (rainfall / 1500)
        + 0.15 * (yield_amount / yield_amount.max())
        - 0.10 * (previous_loans / 5)          # more prior loans → slight risk
        + np.random.normal(0, 0.08, n_samples)  # real-world noise
    )

    # Sigmoid-style threshold: probability > 0.50 → approved
    prob_approved = 1 / (1 + np.exp(-8 * (score - 0.50)))
    approved = (np.random.rand(n_samples) < prob_approved).astype(int)

    df = pd.DataFrame(
        {
            "farm_size":      farm_size,
            "soil_score":     soil_score,
            "rainfall":       rainfall,
            "previous_loans": previous_loans,
            "yield_amount":   yield_amount,
            #"crop_diversity": crop_diversity,
            "approved":       approved,
        }
    )
    return df


# ── 4. Load / prepare data ────────────────────────────────────────────────────
print("=" * 60)
print("  AgriTrust AI – Agricultural Loan Credit Scorer")
print("=" * 60)

print("\n[1/5] Generating synthe    tic dataset …")
df = generate_dataset(n_samples=600)
print(f"      Dataset shape : {df.shape}")
print(f"      Approval rate : {df['approved'].mean():.2%}")
print(df.describe().round(2))

# Separate features (X) and target (y)
FEATURES = ["farm_size", "soil_score", "rainfall", "previous_loans", "yield_amount"]
TARGET   = "approved"

X = df[FEATURES]
y = df[TARGET]

# ── 5. Train / test split (80 / 20) ──────────────────────────────────────────
print("\n[2/5] Splitting data (80 % train / 20 % test) …")
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=RANDOM_SEED,
    stratify=y,          # preserve class balance in both splits
)
print(f"      Train samples : {len(X_train)}")
print(f"      Test  samples : {len(X_test)}")

# ── 6. Model definition & training ───────────────────────────────────────────
print("\n[3/5] Training XGBoost classifier …")

model = XGBClassifier(
    max_depth       = 4,        # tree depth – controls model complexity
    learning_rate   = 0.05,     # shrinks feature weights; lower → more robust
    n_estimators    = 300,      # number of boosting rounds
    subsample       = 0.8,      # row sampling per tree (reduces over-fitting)
    colsample_bytree= 0.8,      # feature sampling per tree
    use_label_encoder=False,
    eval_metric     = "logloss",
    random_state    = RANDOM_SEED,
    n_jobs          = -1,       # use all available CPU cores
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False,
)
print("      Training complete.")

# ── 7. Evaluation ─────────────────────────────────────────────────────────────
print("\n[4/5] Evaluating model on test set …")

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]   # probability of class 1

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_pred_prob)

print("\n" + "─" * 45)
print(f"  Accuracy      : {accuracy:.4f}  ({accuracy*100:.2f} %)")
print(f"  ROC-AUC Score : {roc_auc:.4f}")
print("─" * 45)
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Rejected (0)", "Approved (1)"]))

# ── 8. Feature importance (informational) ────────────────────────────────────
importance_df = (
    pd.Series(model.feature_importances_, index=FEATURES)
    .sort_values(ascending=False)
    .reset_index()
)
importance_df.columns = ["Feature", "Importance"]
print("  Feature Importances:")
print(importance_df.to_string(index=False))

# ── 9. Save the trained model ─────────────────────────────────────────────────
print("\n[5/5] Saving model …")

MODEL_DIR  = "model"
MODEL_PATH = os.path.join(MODEL_DIR, "credit_model.pkl")

# Create the model/ directory if it doesn't already exist
os.makedirs(MODEL_DIR, exist_ok=True)

joblib.dump(model, MODEL_PATH)
print(f"      Model saved to : {MODEL_PATH}")

print("\n" + "=" * 60)
print("  Done! AgriTrust AI credit scorer is ready.")
print("=" * 60)