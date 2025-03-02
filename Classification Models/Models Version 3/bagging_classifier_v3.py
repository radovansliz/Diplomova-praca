import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from joblib import dump, load
import os

# Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# Odstránenie stĺpcov "Hash" a "Category"
df = df.drop(columns=["Hash", "Category"], errors='ignore')
print("Columns 'Hash' and 'Category' removed.")

# Definovanie vstupných a výstupných hodnôt
X = df.drop(columns=["Family"])
y = df["Family"]

# Rozdelenie datasetu na tréningovú, testovaciu a validačnú množinu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Cesta k uloženému modelu
model_path = "bc_model.joblib"

# Kontrola, či existuje uložený model
if os.path.exists(model_path):
    print("Načítavam uložený model...")
    bc_model = load(model_path)
else:
    # Definícia základného klasifikátora
    base_estimator = DecisionTreeClassifier(max_depth=8, min_samples_split=4, random_state=42)

    # Definícia Bagging Classifier
    bc_model = BaggingClassifier(
        estimator=base_estimator,
        n_estimators=75,
        max_samples=0.6,
        max_features=0.6,
        bootstrap=True,
        bootstrap_features=False,
        random_state=42
    )

    # Tréning modelu
    bc_model.fit(X_train, y_train)

    # Uloženie modelu
    dump(bc_model, model_path)
    print(f"Model bol natrenovaný a uložený ako: {model_path}")

# Hodnotenie na validačnej množine
y_val_pred = bc_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Validation Accuracy:", val_accuracy)

# Krížová validácia
cv_scores = cross_val_score(bc_model, X_train, y_train, cv=5)
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Testovanie modelu
y_test_pred = bc_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Results:")
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", test_accuracy)

# Funkcia na vyhodnotenie modelu
def evaluate_model(model, X_val, y_val, X_test, y_test):
    # 1. Uloženie konfúznej matice
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_bc.png")
    plt.close()

    # 2. Uloženie textového reportu (klasifikačná správa)
    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_bc.txt", "w") as f:
        f.write("Validation Accuracy: {:.2f}\n".format(val_accuracy))
        f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
        f.write("\nCross-Validation Scores: {}\n".format(cv_scores))
        f.write("Mean CV Score: {:.2f}\n".format(cv_scores.mean()))
        f.write("\nClassification Report:\n")
        f.write(report)

    # 3. Learning curve
    train_sizes, train_scores, val_scores = learning_curve(
        model, X_train, y_train, cv=5, train_sizes=np.linspace(0.1, 1.0, 10), scoring='accuracy'
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_scores_mean, label="Train Accuracy")
    plt.plot(train_sizes, val_scores_mean, label="Validation Accuracy")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("learning_curve_bc.png")
    plt.close()

# Zavolanie funkcie na vyhodnotenie
evaluate_model(bc_model, X_val, y_val, X_test, y_test)
print("Model evaluation completed.")
