import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error, r2_score

# Načítanie datasetu
print("🔄 Načítavam dataset...")
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# Zahodenie nepotrebných stĺpcov
print("🛠️ Odstraňujem nepotrebné stĺpce ('Hash' a 'Category')...")
df = df.drop(columns=["Hash", "Category"])

# Definovanie vstupných a výstupných hodnôt
print("📌 Rozdeľujem dataset na vstupné hodnoty (X) a výstupné hodnoty (y)...")
X = df.drop(columns=["Family"])
y = df["Family"]

# Prevod stringových hodnôt y na číselné hodnoty
print("🔢 Konvertujem kategórie 'Family' na číselné hodnoty pomocou LabelEncoder...")
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)  

# Rozdelenie datasetu na tréningovú, testovaciu a validačnú množinu
print("📊 Rozdeľujem dataset na tréningovú, validačnú a testovaciu množinu...")
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42, stratify=y_temp)

# Definícia modelu Support Vector Machine (SVM)
print("🚀 Definujem SVM model s RBF jadrom...")
svm_model = SVC(
    kernel='rbf',          
    C=1.0,                 
    gamma='scale',         
    random_state=42,       
    probability=True       
)

# Tréning modelu
print("🧠 Trénujem SVM model...")
svm_model.fit(X_train, y_train)
print("✅ Tréning dokončený!")

# Hodnotenie na validačnej množine
print("🔍 Vyhodnocujem model na validačnej množine...")
y_val_pred = svm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
print("✅ Validácia dokončená!")
print("📊 Validation Results:")
print(classification_report(y_val, y_val_pred))
print(f"✅ Validation Accuracy: {val_accuracy:.4f}")

# 10-násobná krížová validácia
print("⏳ Spúšťam 10-násobnú krížovú validáciu...")
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=10)
print("✅ Krížová validácia dokončená!")
print("📊 Cross-Validation Scores:", cv_scores)
print(f"📈 Mean CV Score: {cv_scores.mean():.4f}")

# Testovanie modelu
print("🛠️ Testujem model na testovacej množine...")
y_test_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
print("✅ Testovanie dokončené!")
print("📊 Test Results:")
print(classification_report(y_test, y_test_pred))
print(f"✅ Test Accuracy: {test_accuracy:.4f}")

# Výpočet regresných metrík
print("📏 Počítam regresné metriky (MSE, RMSE, R² Score)...")
mse = mean_squared_error(y_test, y_test_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_test_pred)
print(f"📉 MSE: {mse:.4f}")
print(f"📉 RMSE: {rmse:.4f}")
print(f"📉 R² Score: {r2:.4f}")

# Funkcia na vyhodnotenie modelu
def evaluate_model(model, X_val, y_val, X_test, y_test):
    print("📊 Ukladám konfúznu maticu...")
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=le.classes_)
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_svm.png")
    plt.close()
    print("✅ Konfúzna matica uložená!")

    print("📜 Ukladám klasifikačnú správu do súboru...")
    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_svm.txt", "w") as f:
        f.write(f"Validation Accuracy: {val_accuracy:.4f}\n")
        f.write(f"Test Accuracy: {test_accuracy:.4f}\n")
        f.write("\nCross-Validation Scores: {}\n".format(cv_scores))
        f.write(f"Mean CV Score: {cv_scores.mean():.4f}\n")
        f.write("\nClassification Report:\n")
        f.write(report)
        f.write("\nRegression Metrics:\n")
        f.write(f"MSE: {mse:.4f}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
    print("✅ Klasifikačná správa uložená!")

    print("📈 Vytváram graf presnosti pri rôznych veľkostiach tréningovej množiny...")
    train_sizes = np.linspace(0.1, 0.9, 10)  # OPRAVA: už nezahŕňa 1.0
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
    print("✅ Graf presnosti uložený!")

# Zavolanie funkcie na vyhodnotenie
print("📊 Spúšťam komplexnú evaluáciu modelu...")
evaluate_model(svm_model, X_val, y_val, X_test, y_test)
print("🎯 Evaluácia dokončená!")
