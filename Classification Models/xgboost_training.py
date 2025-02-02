import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)
print("Dataset nacitany")

# Kontrola na chýbajúce hodnoty a predspracovanie
if df.isnull().sum().sum() > 0:
    print("Chýbajúce hodnoty zistené. Nahrádzam mediánom...")
    df = df.fillna(df.median())

# Definovanie vstupných a výstupných hodnôt
X = df.drop(columns=["Family", "Hash", "Category"])
y = df["Family"]

print("X a Y inicializovane")

# Konverzia cieľovej premennej pomocou LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

print("Label encoding completed. Classes:", label_encoder.classes_)

# Rozdelenie datasetu na tréningovú, testovaciu a validačnú množinu
# pôvodný kód
# X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
# X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp)

# upravený kód
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print("Dataset rozdeleny")

# Vytvorenie DMatrix pre XGBoost
# pôvodný kód
# dtrain = xgb.DMatrix(X_train, label=y_train)
# dval = xgb.DMatrix(X_val, label=y_val)
# dtest = xgb.DMatrix(X_test, label=y_test)

# upravený kód
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)
dtest = xgb.DMatrix(X_test, label=y_test)

# Definícia parametrov pre XGBoost
param = {
    "objective": "multi:softmax",  # Multiclass classification
    "num_class": len(np.unique(y)),  # Počet tried
    # pôvodné nastavenie max_depth
    # "max_depth": 6,
    # upravené nastavenie max_depth
    "max_depth": 4,  # Zníženie hĺbky stromu pre lepšiu generalizáciu
    "learning_rate": 0.1,
    # pridanie regularizácie
    "lambda": 1.0,  # L2 regularizácia
    "alpha": 0.1,   # L1 regularizácia
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "seed": 42,
}

num_round = 100

# Krížová validácia
print("Running cross-validation...")
cv_results = xgb.cv(
    param,
    dtrain,
    num_boost_round=num_round,
    nfold=10,
    metrics=["mlogloss", "merror"],
    early_stopping_rounds=20,
    seed=42
)

optimal_rounds = cv_results['test-mlogloss-mean'].idxmin() + 1
print(f"Optimal number of boosting rounds: {optimal_rounds}")

print("Cross-validation results:")
print(cv_results)

# Tréning modelu
print("Training XGBoost model...")
bst = xgb.train(
    param,
    dtrain,
    # pôvodné nastavenie num_boost_round
    # num_boost_round=optimal_rounds,
    # upravené
    num_boost_round=optimal_rounds,  # Optimalizované podľa CV výsledkov
    evals=[(dval, "Validation")],
    early_stopping_rounds=10
)

# Hodnotenie na validačnej množine
y_val_pred = bst.predict(dval).astype(int)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Validation Accuracy:", val_accuracy)

# Testovanie modelu
y_test_pred = bst.predict(dtest).astype(int)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("Test Results:")
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", test_accuracy)

# Vizualizácia dôležitosti vlastností
print("Saving feature importance plot...")
plt.figure(figsize=(16, 12))
xgb.plot_importance(bst, importance_type="weight")
plt.title("Feature Importance")
plt.savefig("feature_importance_xgboost.png")  # Uloženie grafu
plt.close()

# Funkcia na vyhodnotenie modelu
def evaluate_model(model, dval, y_val, dtest, y_test):
    # 1. Uloženie konfúznej matice
    cm = confusion_matrix(y_test, model.predict(dtest).astype(int))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_xgboost.png")  # Uloženie grafu
    plt.close()

    # 2. Uloženie textového reportu (klasifikačná správa)
    report = classification_report(y_test, model.predict(dtest).astype(int))
    with open("classification_report_xgboost.txt", "w") as f:
        # Základné informácie o modeli
        f.write("XGBoost Model Parameters:\n")
        for key, value in param.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Number of Boosting Rounds: {num_round}\n\n")
        
        # Informácie o krížovej validácii
        f.write("Cross-Validation Settings:\n")
        f.write(f"Number of Folds: 10\n")
        f.write(f"Early Stopping Rounds: 10\n\n")
        f.write("Cross-Validation Results:\n")
        f.write(cv_results.to_string() + "\n\n")

        # Výsledky hodnotenia
        f.write("Validation Accuracy: {:.2f}\n".format(val_accuracy))
        f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
        f.write("\nClassification Report:\n")
        f.write(report)

    # 3. Loss krivka - simulujeme presnosť pri rôznych veľkostiach tréningovej množiny
    train_sizes = np.linspace(0.1, 0.9, 9)
    train_scores, val_scores = [], []
    for frac in train_sizes:
        min_samples = 50  # Minimálny počet vzoriek
        # pôvodné rozdelenie
        # X_frac, _, y_frac, _ = train_test_split(
        #     X_train, y_train, train_size=float(frac), stratify=y_train, random_state=42
        # )
        # upravené
        X_frac, _, y_frac, _ = train_test_split(
            X_train, y_train, train_size=max(float(frac), min_samples / len(X_train)),
            stratify=y_train, random_state=42
        )
        dfrac = xgb.DMatrix(X_frac, label=y_frac)
        model = xgb.train(param, dfrac, num_boost_round=optimal_rounds)
        train_scores.append(accuracy_score(y_frac, model.predict(dfrac).astype(int)))
        val_scores.append(accuracy_score(y_val, bst.predict(dval).astype(int)))

    plt.plot(train_sizes, train_scores, label="Train Accuracy")
    plt.plot(train_sizes, val_scores, label="Validation Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Training Set Size Fraction")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("accuracy_curve_xgboost.png")  # Uloženie grafu
    plt.close()


# Zavolanie funkcie na vyhodnotenie
evaluate_model(bst, dval, y_val, dtest, y_test)
