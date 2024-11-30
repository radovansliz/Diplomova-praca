import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# Definovanie vstupných a výstupných hodnôt
X = df.drop(columns=["Family"])
y = df["Family"]

# Rozdelenie datasetu na tréningovú, testovaciu a validačnú množinu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Definícia modelu Support Vector Machine (SVM)
svm_model = SVC(
    kernel='rbf',          # Jadrová funkcia, napr. 'linear', 'poly', 'rbf', 'sigmoid'
    C=1.0,                 # Regularizačný parameter (hyperparameter na ladenie)
    gamma='scale',         # Parameter gamma (len pre rbf, poly, sigmoid jadrá)
    random_state=42,       # Pre reproducibilitu
    probability=True       # Ak potrebujeme pravdepodobnosti (napr. pre ROC krivky)
)

# Tréning modelu
svm_model.fit(X_train, y_train)

# Hodnotenie na validačnej množine
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Validation Accuracy:", val_accuracy)

# Krížová validácia
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)  # 5-násobná krížová validácia
print("Cross-Validation Scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Testovanie modelu
y_test_pred = svm_model.predict(X_test)
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
    plt.savefig("confusion_matrix_svm.png")
    plt.close()

    # 2. Uloženie textového reportu (klasifikačná správa)
    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_svm.txt", "w") as f:
        f.write("Validation Accuracy: {:.2f}\n".format(val_accuracy))
        f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
        f.write("\nCross-Validation Scores: {}\n".format(cv_scores))
        f.write("Mean CV Score: {:.2f}\n".format(cv_scores.mean()))
        f.write("\nClassification Report:\n")
        f.write(report)

    # 3. Loss krivka - simulujeme presnosť pri rôznych veľkostiach tréningovej množiny
    train_sizes = np.linspace(0.1, 1.0, 10)
    train_scores, val_scores = [], []
    for frac in train_sizes:
        X_frac, _, y_frac, _ = train_test_split(X_train, y_train, train_size=frac, stratify=y_train, random_state=42)
        model.fit(X_frac, y_frac)
        train_scores.append(accuracy_score(y_frac, model.predict(X_frac)))
        val_scores.append(accuracy_score(y_val, model.predict(X_val)))

    plt.plot(train_sizes, train_scores, label="Train Accuracy")
    plt.plot(train_sizes, val_scores, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Training Set Size Fraction")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve_svm.png")
    plt.close()


# Zavolanie funkcie na vyhodnotenie
evaluate_model(svm_model, X_val, y_val, X_test, y_test)
