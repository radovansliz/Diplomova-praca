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

#Ulozenie X_train mnoziny na nasledne pouzitie vysvetlitelnych metod
X_train.to_csv("X_train_adaboost.csv", index=False)

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



    # Tréning modelu
    adaboost_model.fit(X_train, y_train)

    # Uloženie modelu
    dump(adaboost_model, model_path)
    print(f"Model bol natrenovaný a uložený ako: {model_path}")

# Hodnotenie na validačnej množine
y_val_pred = adaboost_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Accuracy:", val_accuracy)

# Výpočet metrik pre validačnú množinu
cm_val = confusion_matrix(y_val, y_val_pred)
FP_val = cm_val.sum(axis=0) - np.diag(cm_val)
FN_val = cm_val.sum(axis=1) - np.diag(cm_val)
TP_val = np.diag(cm_val)
TN_val = cm_val.sum() - (FP_val + FN_val + TP_val)

FPR_val = FP_val / (FP_val + TN_val)

# Krížová validácia
cv_scores = cross_val_score(adaboost_model, X_train, y_train, cv=10)
mean_cv_score = cv_scores.mean()
print("Mean CV Score:", mean_cv_score)

# Testovanie modelu
y_test_pred = adaboost_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Results:")
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", test_accuracy)

# Výpočet metrik pre testovaciu množinu
cm_test = confusion_matrix(y_test, y_test_pred)
FP_test = cm_test.sum(axis=0) - np.diag(cm_test)
FN_test = cm_test.sum(axis=1) - np.diag(cm_test)
TP_test = np.diag(cm_test)
TN_test = cm_test.sum() - (FP_test + FN_test + TP_test)
FPR_test = FP_test / (FP_test + TN_test)


# Funkcia na vyhodnotenie modelu
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # 1. Uloženie konfúznej matice
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_adaboost.png")
    plt.close()

    # 2. Uloženie textového reportu (klasifikačná správa)
    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_adaboost.txt", "w") as f:
        f.write("Validation Accuracy: {:.2f}\n".format(val_accuracy))
        f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
        f.write("\nClassification Report:\n")
        f.write(report)

        # Uloženie výsledkov krížovej validácie
        f.write("\n--- Výsledky 10-násobnej krížovej validácie ---\n")
        f.write("Presnosti pre jednotlivé foldy: {}\n".format(cv_scores))
        f.write("Priemerná presnosť: {:.2f}\n".format(np.mean(cv_scores)))
        f.write("Štandardná odchýlka: {:.2f}\n".format(np.std(cv_scores)))

        # Výpočtové hodnoty
        f.write("\n--- Výpočtové hodnoty pre validačnú množinu ---\n")
        f.write(f"FP_val: {FP_val}\nFN_val: {FN_val}\nTP_val: {TP_val}\nTN_val: {TN_val}\nFPR_val: {FPR_val}\n")
        f.write("Priemerný Validation FPR: {:.2f}\n".format(np.mean(FPR_val)))

        f.write("\n--- Výpočtové hodnoty pre testovaciu množinu ---\n")
        f.write(f"FP_test: {FP_test}\nFN_test: {FN_test}\nTP_test: {TP_test}\nTN_test: {TN_test}\nFPR_test: {FPR_test}\n")
        f.write("Priemerný Test FPR: {:.2f}\n".format(np.mean(FPR_test)))

    # 3. Learning curve
    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=10, scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, label="Train Accuracy")
    plt.plot(train_sizes, val_mean, label="Validation Accuracy")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("learning_curve_adaboost.png")
    plt.close()

# Zavolanie funkcie na vyhodnotenie
evaluate_model(adaboost_model, X_train, y_train, X_test, y_test)