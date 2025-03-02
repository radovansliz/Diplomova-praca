import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load
import os

# Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# Kontrola na chýbajúce hodnoty a predspracovanie
if df.isnull().sum().sum() > 0:
    df = df.fillna(df.median())

# Definovanie vstupných a výstupných hodnôt
X = df.drop(columns=["Family", "Hash", "Category"])
y = df["Family"]

# Rozdelenie datasetu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp)

# Cesta k uloženému modelu
model_path = "adaboost_model.joblib"

# Kontrola, či existuje uložený model
if os.path.exists(model_path):
    print("Načítavam uložený model...")
    adaboost_model = load(model_path)
else:
    # Definícia modelu AdaBoost
    adaboost_model = AdaBoostClassifier(
        estimator=DecisionTreeClassifier(max_depth=3, random_state=42),
        n_estimators=100,
        learning_rate=0.1,
        random_state=42
    )

    # Krížová validácia
    cv_scores = cross_val_score(adaboost_model, X_train, y_train, cv=10, scoring="accuracy")
    mean_cv_score = cv_scores.mean()
    print("Mean CV Score:", mean_cv_score)

    # Tréning modelu
    adaboost_model.fit(X_train, y_train)

    # Uloženie modelu
    dump(adaboost_model, model_path)
    print(f"Model bol natrenovaný a uložený ako: {model_path}")

# Hodnotenie na validačnej množine
y_val_pred = adaboost_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Uloženie výsledkov do súboru
with open("model_results_adaboost.txt", "w") as f:
    f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_val, y_val_pred))

# Learning Curve
train_sizes, train_scores, val_scores = learning_curve(
    adaboost_model, X_train, y_train, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_mean, label='Training Accuracy')
plt.plot(train_sizes, val_mean, label='Validation Accuracy')
plt.title("Learning Curve of AdaBoost")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("learning_curve_adaboost.png")
plt.close()
