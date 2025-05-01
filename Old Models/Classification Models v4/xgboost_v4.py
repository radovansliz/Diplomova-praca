import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp)

#Ulozenie X_train mnoziny na nasledne pouzitie vysvetlitelnych metod
X_train.to_csv("X_train_xgboost.csv", index=False)
y_train.to_csv("y_train_xgboost.csv", index=False)
X_test.to_csv("X_test_xgboost.csv", index=False)
y_test.to_csv("y_test_xgboost.csv", index=False)
print("Train and test datasets saved for explainability methods.")

# Cesta k uloženému modelu
model_path = "xgboost_model.joblib"

# 5. Definícia a tréning modelu pomocou XGBClassifier
if os.path.exists(model_path):
    print("Načítavam uložený model...")
    xgb_clf = load(model_path)
else:
    xgb_clf = XGBClassifier(
        objective="multi:softmax",
        num_class=len(np.unique(y)),
        max_depth=2,
        learning_rate=0.05,
        reg_lambda=3,
        reg_alpha=1,
        subsample=0.8,
        colsample_bytree=0.7,
        seed=42,
        n_estimators=120  # Počet boosting rounds
    )
    xgb_clf.fit(X_train, y_train)
    dump(xgb_clf, model_path)
    print(f"Model bol natrenovaný a uložený ako: {model_path}")

# 6. Hodnotenie na validačnej množine
y_val_pred = xgb_clf.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Výpočet metrik pre validačnú množinu
cm_val = confusion_matrix(y_val, y_val_pred)
FP_val = cm_val.sum(axis=0) - np.diag(cm_val)
FN_val = cm_val.sum(axis=1) - np.diag(cm_val)
TP_val = np.diag(cm_val)
TN_val = cm_val.sum() - (FP_val + FN_val + TP_val)
FPR_val = FP_val / (FP_val + TN_val)

# 7. Krížová validácia
cv_scores = cross_val_score(xgb_clf, X_train, y_train, cv=10)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# 8. Vyhodnotenie na testovacej množine
y_test_pred = xgb_clf.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("\nTest Accuracy:", test_accuracy)

# Výpočet metrik pre testovaciu množinu
cm_test = confusion_matrix(y_test, y_test_pred)
FP_test = cm_test.sum(axis=0) - np.diag(cm_test)
FN_test = cm_test.sum(axis=1) - np.diag(cm_test)
TP_test = np.diag(cm_test)
TN_test = cm_test.sum() - (FP_test + FN_test + TP_test)
FPR_test = FP_test / (FP_test + TN_test)

# 9. Funkcia na vyhodnotenie modelu
def evaluate_model(model, X_train, y_train, X_test, y_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_xgboost.png")
    plt.close()

    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_xgboost.txt", "w") as f:
        f.write("Validation Accuracy: {:.2f}\n".format(val_accuracy))
        f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
        f.write("\nClassification Report:\n")
        f.write(report)

        f.write("\n--- Výsledky 10-násobnej krížovej validácie ---\n")
        f.write("Presnosti pre jednotlivé foldy: {}\n".format(cv_scores))
        f.write("Priemerná presnosť: {:.2f}\n".format(np.mean(cv_scores)))
        f.write("Štandardná odchýlka: {:.2f}\n".format(np.std(cv_scores)))

        f.write("\n--- Výpočtové hodnoty pre validačnú množinu ---\n")
        f.write(f"FP_val: {FP_val}\nFN_val: {FN_val}\nTP_val: {TP_val}\nTN_val: {TN_val}\nFPR_val: {FPR_val}\n")
        f.write("Priemerný Validation FPR: {:.2f}\n".format(np.mean(FPR_val)))

        f.write("\n--- Výpočtové hodnoty pre testovaciu množinu ---\n")
        f.write(f"FP_test: {FP_test}\nFN_test: {FN_test}\nTP_test: {TP_test}\nTN_test: {TN_test}\nFPR_test: {FPR_test}\n")
        f.write("Priemerný Test FPR: {:.2f}\n".format(np.mean(FPR_test)))

    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, label="Train Accuracy")
    plt.plot(train_sizes, val_mean, label="Validation Accuracy")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("learning_curve_xgboost.png")
    plt.close()

# Zavolanie funkcie na vyhodnotenie
evaluate_model(xgb_clf, X_train, y_train, X_test, y_test)
print("Model evaluation completed.")
print("Results saved to results.txt, confusion_matrix_xgboost.png, and learning_curve_xgboost.png")
