import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from joblib import dump, load
import os

# 1. Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# 2. Predspracovanie dát
X = df.drop(columns=["Family", "Hash", "Category"])  
y = df["Family"]

# 3. Label Encoding
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# 4. Rozdelenie dát
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 5. Vytvorenie XGBoost DMatrix
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

# Cesta k uloženému modelu
model_path = "xgboost_model.joblib"

# 6. Definícia parametrov modelu
param = {
    "objective": "multi:softmax",
    "num_class": len(np.unique(y)),  
    "max_depth": 2,
    "learning_rate": 0.05,
    "lambda": 3,
    "alpha": 1,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "seed": 42,
}

num_round = 120

# Kontrola, či existuje uložený model
if os.path.exists(model_path):
    print("Načítavam uložený model...")
    bst = load(model_path)
else:
    # 7. 10-násobná krížová validácia
    cv_results = xgb.cv(
        param,
        dtrain,
        num_boost_round=num_round,
        nfold=10,
        metrics=["mlogloss", "merror"],
        early_stopping_rounds=20,
        seed=42
    )

    # 8. Zistenie optimálneho počtu boosting rounds
    optimal_rounds = cv_results['test-mlogloss-mean'].idxmin() + 1
    print(f"Optimal number of boosting rounds: {optimal_rounds}")

    # 9. Tréning finálneho modelu
    bst = xgb.train(param, dtrain, num_boost_round=optimal_rounds)

    # Uloženie modelu
    dump(bst, model_path)
    print(f"Model bol natrenovaný a uložený ako: {model_path}")

# 10. Vyhodnotenie na testovacej množine
y_test_pred = bst.predict(dtest).astype(int)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("\nTest Accuracy:", test_accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# 11. Konfúzna matica
cm = confusion_matrix(y_test, y_test_pred)
plt.figure(figsize=(6, 5))
plt.imshow(cm, cmap='Blues', interpolation='nearest')
plt.title("Confusion Matrix")
plt.colorbar()
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.savefig("confusion_matrix_xgboost.png")
plt.close()

# 12. Learning Curve (Trénovacia a validačná presnosť pri rôznych veľkostiach datasetu)
train_sizes = np.linspace(0.1, 0.9, 9)
train_scores, val_scores = [], []

for frac in train_sizes:
    X_frac, _, y_frac, _ = train_test_split(
        X_train, y_train, train_size=frac, stratify=y_train, random_state=42
    )
    dfrac = xgb.DMatrix(X_frac, label=y_frac)
    model = xgb.train(param, dfrac, num_boost_round=optimal_rounds)
    train_scores.append(accuracy_score(y_frac, model.predict(dfrac).astype(int)))
    val_scores.append(accuracy_score(y_test, bst.predict(dtest).astype(int)))

# 13. Vizualizácia Learning Curve
plt.plot(train_sizes, train_scores, label="Train Accuracy")
plt.plot(train_sizes, val_scores, label="Validation Accuracy")
plt.title("Learning Curve")
plt.xlabel("Training Set Size Fraction")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("learning_curve_xgboost.png")
plt.close()

# 14. Uloženie výsledkov do súboru
with open("results.txt", "w") as f:
    f.write("===== XGBoost Model Results =====\n\n")
    f.write("Hyperparameters:\n")
    for key, value in param.items():
        f.write(f"{key}: {value}\n")
    f.write(f"\nTest Accuracy: {test_accuracy:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_test_pred) + "\n")
    f.write("\n===== Learning Curve Data =====\n")
    for i in range(len(train_sizes)):
        f.write(f"Train Size: {train_sizes[i]:.2f}, Train Accuracy: {train_scores[i]:.4f}, Validation Accuracy: {val_scores[i]:.4f}\n")

print("Results saved to results.txt, confusion_matrix_xgboost.png, and learning_curve_xgboost.png")
