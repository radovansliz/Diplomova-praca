import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from lightgbm import LGBMClassifier,early_stopping, log_evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score, log_loss

# Načítanie datasetu
data_path = "12k_samples_12_families.csv"
df = pd.read_csv(data_path)

# Odstránenie nežiaducich stĺpcov
df = df.drop(columns=["Hash", "Category"], errors='ignore')

# Definovanie vstupných a výstupných hodnôt
X = df.drop(columns=["Family"])
y = df["Family"]

# Rozdelenie datasetu
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Definícia modelu LightGBM
lgbm_model = LGBMClassifier(
    boosting_type='gbdt',
    n_estimators=75,
    num_leaves = 5,
    max_depth=4,
    learning_rate=0.02,
    subsample=0.7,
    colsample_bytree=0.8,
    reg_lambda=1.2,
    reg_alpha=1.2,
    min_child_samples=60,
    random_state=42,
    bagging_fraction=0.7,
    bagging_freq=5
)

# Tréning modelu s early stopping pomocou callbacks
print("Training the model with Early Stopping...")
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # Validačná množina
    eval_metric='logloss',  # Sledovanie log loss metriky
    callbacks=[early_stopping(stopping_rounds=10), log_evaluation(5)]  # Early stopping a výpis každých 10 iterácií
)
print("Training complete with Early Stopping!")

# Krížová validácia
print("Performing 10-fold cross-validation...")
cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=10, scoring='f1_weighted')
print("Cross-validation scores:", cv_scores)
print("Mean CV Score:", cv_scores.mean())

# Hodnotenie na validačnej množine
y_val_pred = lgbm_model.predict(X_val)
val_accuracy = accuracy_score(y_val, y_val_pred)
val_f1 = f1_score(y_val, y_val_pred, average='weighted')
print("Validation Results:")
print(classification_report(y_val, y_val_pred))
print("Validation Accuracy:", val_accuracy)
print("Validation F1 Score:", val_f1)

# Testovanie modelu
y_test_pred = lgbm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred, average='weighted')
print("Test Results:")
print(classification_report(y_test, y_test_pred))
print("Test Accuracy:", test_accuracy)
print("Test F1 Score:", test_f1)

# Uloženie výsledkov do súboru
with open("classification_report_lgbm.txt", "w") as f:
    f.write("Model Configuration:\n")
    f.write(str(lgbm_model.get_params()))
    f.write("\n\nValidation Accuracy: {:.2f}\n".format(val_accuracy))
    f.write("Validation F1 Score: {:.2f}\n".format(val_f1))
    f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
    f.write("Test F1 Score: {:.2f}\n".format(test_f1))
    f.write("\nCross-Validation Scores: {}\n".format(cv_scores))
    f.write("Mean CV Score: {:.2f}\n".format(cv_scores.mean()))
    f.write("\nClassification Report:\n")
    f.write(classification_report(y_test, y_test_pred))

# Uloženie konfúznej matice
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix_lgbm.png")
plt.close()

# Learning Curve (10-fold cross-validation)
train_sizes, train_scores, val_scores = learning_curve(lgbm_model, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 10))
train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_scores_mean, label="Training Score")
plt.plot(train_sizes, val_scores_mean, label="Validation Score")
plt.title("Learning Curve (10-Fold CV)")
plt.xlabel("Training Examples")
plt.ylabel("Accuracy")
plt.legend()
plt.savefig("learning_curve_lgbm.png")
plt.close()

# Neg Log Loss Curve
y_proba = lgbm_model.predict_proba(X_test)
log_loss_value = log_loss(y_test, y_proba)

plt.figure()
plt.plot(np.arange(len(y_test)), -np.log(y_proba.max(axis=1)), label="Negative Log Loss")
plt.title("Negative Log Loss Curve")
plt.xlabel("Test Sample Index")
plt.ylabel("-Log Loss")
plt.legend()
plt.savefig("neg_log_loss_curve_lgbm.png")
plt.close()

print("Results saved to classification_report_lgbm.txt, confusion_matrix_lgbm.png, learning_curve_lgbm.png, and neg_log_loss_curve.png")
