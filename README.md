## Suraj's Tasks
# Tasks Completed — Predictive Pulse

---


## Task 1: Data Splitting

Split the processed dataset (`patient_data_processed.csv`, 1 825 records) into training and testing sets.

| Parameter | Value |
|-----------|-------|
| Split ratio | 80% train / 20% test |
| `random_state` | 42 |
| `stratify` | `y` (Stages) |
| Training samples | 1 460 |
| Testing samples | 365 |

Class distribution is preserved across both sets:

| Stage | Train | Test |
|-------|-------|------|
| 0 — Normal | 269 | 67 |
| 1 — Hypertension Stage-1 | 518 | 130 |
| 2 — Hypertension Stage-2 | 480 | 120 |
| 3 — Hypertensive Crisis | 193 | 48 |

---

## Task 2: Model Persistence

After training and comparing 8 classifiers, the best-performing model was saved using `joblib`.

### With BP Features (Systolic + Diastolic included)

| Best Model | Test Accuracy | Saved File |
|------------|---------------|------------|
| **Decision Tree** | **100%** | `best_model_with_bp.pkl` |

### Without BP Features (demographics + symptoms only)

| Best Model | Test Accuracy | Saved File |
|------------|---------------|------------|
| **RBF SVM** | **79.18%** | `best_model_without_bp.pkl` |

### Saved Artifacts

| File | Description |
|------|-------------|
| `best_model_with_bp.pkl` | Best model trained with BP features |
| `best_model_without_bp.pkl` | Best model trained without BP features |
| `model_features_with_bp.pkl` | Feature list (12 features) |
| `model_features_without_bp.pkl` | Feature list (10 features) |

### Loading the Saved Model

```python
import joblib

model = joblib.load("best_model_without_bp.pkl")
features = joblib.load("model_features_without_bp.pkl")
prediction = model.predict(new_data[features])
```