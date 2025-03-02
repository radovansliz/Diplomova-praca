import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# Kontrola na chýbajúce hodnoty a predspracovanie
if df.isnull().sum().sum() > 0:
    print("Chýbajúce hodnoty zistené. Nahrádzam mediánom...")
    df = df.fillna(df.median())

# Definovanie vstupných a výstupných hodnôt
X = df.drop(columns=["Family", "Hash", "Category"])
y = df["Family"]

# Rozdelenie datasetu na tréningovú, testovaciu a validačnú množinu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Definícia modelu AdaBoost
adaboost_model = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1, random_state=42),  # Základný model (Decision Stump)
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)

# Krížová validácia s 10 foldami
cv_scores = cross_val_score(adaboost_model, X_train, y_train, cv=10, scoring="accuracy")
print("Cross-Validation Scores (10-fold):", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Optimalizácia hyperparametrov pomocou GridSearchCV
param_grid = {
    "n_estimators": [50, 100, 150],
    "learning_rate": [0.01, 0.1, 1.0]
}

grid_search = GridSearchCV(
    estimator=AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=42), random_state=42),
    param_grid=param_grid,
    cv=10,
    scoring="accuracy",
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

# Najlepšie parametre a skóre
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Tréning modelu s najlepšími parametrami
best_adaboost_model = grid_search.best_estimator_
best_adaboost_model.fit(X_train, y_train)

# Hodnotenie na validačnej množine
y_val_pred = best_adaboost_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Validation Accuracy:", val_accuracy)

# Testovanie modelu
y_test_pred = best_adaboost_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Results:")
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", test_accuracy)

# Vizualizácia dôležitosti vlastností
feature_importances = best_adaboost_model.feature_importances_
plt.figure(figsize=(16, 12))
plt.bar(range(len(feature_importances)), feature_importances)
plt.title("Feature Importance")
plt.xlabel("Features")
plt.ylabel("Importance")
plt.savefig("feature_importance_adaboost.png")
plt.close()

# Funkcia na vyhodnotenie modelu
def evaluate_model(model, X_val, y_val, X_test, y_test):
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
        f.write("Validation Accuracy: {:.2f}\n".format(accuracy_score(y_val, model.predict(X_val))))
        f.write("Test Accuracy: {:.2f}\n".format(accuracy_score(y_test, model.predict(X_test))))
        f.write("\nClassification Report:\n")
        f.write(report)

    # 3. Loss krivka - simulujeme presnosť pri rôznych veľkostiach tréningovej množiny
    train_sizes = np.linspace(0.1, 0.99, 10)  # Upravený rozsah na max 0.99
    train_scores, val_scores = [], []
    for frac in train_sizes:
        # Explicitná konverzia na float a zabezpečenie správneho rozsahu
        X_frac, _, y_frac, _ = train_test_split(X_train, y_train, train_size=float(frac), stratify=y_train, random_state=42)
        model.fit(X_frac, y_frac)
        train_scores.append(accuracy_score(y_frac, model.predict(X_frac)))
        val_scores.append(accuracy_score(y_val, model.predict(X_val)))

    plt.plot(train_sizes, train_scores, label="Train Accuracy")
    plt.plot(train_sizes, val_scores, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Training Set Size Fraction")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve_adaboost.png")
    plt.close()



# Zavolanie funkcie na vyhodnotenie
evaluate_model(best_adaboost_model, X_val, y_val, X_test, y_test)
