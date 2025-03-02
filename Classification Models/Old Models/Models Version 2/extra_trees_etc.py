import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# Definovanie vstupných a výstupných hodnôt
X = df.drop(columns=["Family", "Hash", "Category"])
y = df["Family"]

# Rozdelenie datasetu na tréningovú, testovaciu a validačnú množinu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp)

# Definícia modelu Extra Trees Classifier
etc_model = ExtraTreesClassifier(
    n_estimators=80,  # Počet stromov (hyperparameter na ladenie)
    max_depth=9,    # Maximálna hĺbka stromov (hyperparameter na ladenie)
    random_state=42,   # Pre reprodukovateľnosť
    min_samples_split=8, # Zabraňuje príliš jemným rozdeleniam
    min_samples_leaf=3,   # Zabráni vzniku príliš malých listov
    max_features='sqrt',  # Použije iba podmnožinu vlastností pre každý strom
)

# Tréning modelu
etc_model.fit(X_train, y_train)

# Hodnotenie na validačnej množine
y_val_pred = etc_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Validation Accuracy:", val_accuracy)

# Testovanie modelu
y_test_pred = etc_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Results:")
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", test_accuracy)

# Funkcia na vyhodnotenie modelu
def evaluate_model(model, X_train, y_train, X_test, y_test):
    # 1. Uloženie konfúznej matice
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_etc.png")
    plt.close()

    # 2. Uloženie textového reportu (klasifikačná správa)
    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_etc.txt", "w") as f:
        f.write("Validation Accuracy: {:.2f}\n".format(val_accuracy))
        f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
        f.write("\nClassification Report:\n")
        f.write(report)

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
    plt.savefig("learning_curve_etc.png")
    plt.close()

# Zavolanie funkcie na vyhodnotenie
evaluate_model(etc_model, X_train, y_train, X_test, y_test)
