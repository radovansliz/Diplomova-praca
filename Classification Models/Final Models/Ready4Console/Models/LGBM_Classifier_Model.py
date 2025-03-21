import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay, f1_score
from joblib import dump, load
import os
import json

def evaluate_model(model, X_train, y_train, X_test, y_test, config, val_accuracy, test_accuracy, cv_scores, FP_val, FN_val, TP_val, TN_val, FPR_val, FP_test, FN_test, TP_test, TN_test, FPR_test):
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.unique(y_test))
    disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix_lgbm.png")
    plt.close()

    report = classification_report(y_test, model.predict(X_test))
    with open("classification_report_lgbm.txt", "w") as f:
        f.write("Validation Accuracy: {:.2f}\n".format(val_accuracy))
        f.write("Test Accuracy: {:.2f}\n".format(test_accuracy))
        f.write("\nClassification Report:\n")
        f.write(report)

        f.write("\n--- Výsledky krížovej validácie ---\n")
        f.write("Presnosti pre jednotlivé foldy: {}\n".format(cv_scores))
        f.write("Priemerná presnosť: {:.2f}\n".format(np.mean(cv_scores)))
        f.write("Štandardná odchýlka: {:.2f}\n".format(np.std(cv_scores)))

        f.write("\n--- Výpočtové hodnoty pre validačnú množinu ---\n")
        f.write(f"FP_val: {FP_val}\nFN_val: {FN_val}\nTP_val: {TP_val}\nTN_val: {TN_val}\nFPR_val: {FPR_val}\n")
        f.write("Priemerný Validation FPR: {:.2f}\n".format(np.mean(FPR_val)))

        f.write("\n--- Výpočtové hodnoty pre testovaciu množinu ---\n")
        f.write(f"FP_test: {FP_test}\nFN_test: {FN_test}\nTP_test: {TP_test}\nTN_test: {TN_test}\nFPR_test: {FPR_test}\n")
        f.write("Priemerný Test FPR: {:.2f}\n".format(np.mean(FPR_test)))

    train_sizes, train_scores, val_scores = learning_curve(model, X_train, y_train, cv=config["cross_validation_n"], scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10))
    train_mean = np.mean(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)

    plt.plot(train_sizes, train_mean, label="Train Accuracy")
    plt.plot(train_sizes, val_mean, label="Validation Accuracy")
    plt.title("Learning Curve")
    plt.xlabel("Training Set Size")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.savefig("learning_curve_lgbm.png")
    plt.close()

def run_lgbm_pipeline(X, y, config, evaluate_flag=True):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=config["test_size_test"], random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=config["test_size_val"], random_state=42, stratify=y_temp)

    X_train.to_csv("X_train_lgbm.csv", index=False)
    y_train.to_csv("y_train_lgbm.csv", index=False)
    X_test.to_csv("X_test_lgbm.csv", index=False)
    y_test.to_csv("y_test_lgbm.csv", index=False)
    print("Train and test datasets saved for explainability methods.")

    model_path = "lgbm_model.joblib"

    if os.path.exists(model_path):
        print("Načítavam uložený model...")
        lgbm_model = load(model_path)
    else:
        lgbm_model = LGBMClassifier(
            boosting_type=config["boosting_type"],
            n_estimators=config["n_estimators"],
            num_leaves=config["num_leaves"],
            max_depth=config["max_depth"],
            learning_rate=config["learning_rate"],
            subsample=config["subsample"],
            colsample_bytree=config["colsample_bytree"],
            reg_lambda=config["reg_lambda"],
            reg_alpha=config["reg_alpha"],
            min_child_samples=config["min_child_samples"],
            bagging_fraction=config["bagging_fraction"],
            bagging_freq=config["bagging_freq"],
            random_state=42
        )

        print("Training the model with Early Stopping...")
        lgbm_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_metric=config["eval_metric"],
            callbacks=[early_stopping(stopping_rounds=config["early_stopping_rounds"]), log_evaluation(5)]
        )
        print("Training complete!")
        dump(lgbm_model, model_path)
        print(f"Model bol natrenovaný a uložený ako: {model_path}")

    print("Performing cross-validation...")
    cv_scores = cross_val_score(lgbm_model, X_train, y_train, cv=config["cross_validation_n"])
    print("Cross-validation scores:", cv_scores)
    print("Mean CV Score:", cv_scores.mean())

    y_val_pred = lgbm_model.predict(X_val)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    print("Validation Results:")
    print(classification_report(y_val, y_val_pred))
    print("Validation Accuracy:", val_accuracy)
    print("Validation F1 Score:", val_f1)

    cm_val = confusion_matrix(y_val, y_val_pred)
    FP_val = cm_val.sum(axis=0) - np.diag(cm_val)
    FN_val = cm_val.sum(axis=1) - np.diag(cm_val)
    TP_val = np.diag(cm_val)
    TN_val = cm_val.sum() - (FP_val + FN_val + TP_val)
    FPR_val = FP_val / (FP_val + TN_val)

    y_test_pred = lgbm_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    print("Test Results:")
    print(classification_report(y_test, y_test_pred))
    print("Test Accuracy:", test_accuracy)
    print("Test F1 Score:", test_f1)

    cm_test = confusion_matrix(y_test, y_test_pred)
    FP_test = cm_test.sum(axis=0) - np.diag(cm_test)
    FN_test = cm_test.sum(axis=1) - np.diag(cm_test)
    TP_test = np.diag(cm_test)
    TN_test = cm_test.sum() - (FP_test + FN_test + TP_test)
    FPR_test = FP_test / (FP_test + TN_test)

    if evaluate_flag:
        evaluate_model(lgbm_model, X_train, y_train, X_test, y_test, config, val_accuracy, test_accuracy, cv_scores, FP_val, FN_val, TP_val, TN_val, FPR_val, FP_test, FN_test, TP_test, TN_test, FPR_test)

if __name__ == "__main__":
    EVALUATE_MODEL = True
    with open(os.path.join("Konfiguracie", "lgbm_config.json"), "r") as config_file:
        config = json.load(config_file)

    data_path = "12k_samples_12_families.csv"
    df = pd.read_csv(data_path)
    df = df.drop(columns=["Hash", "Category"], errors='ignore')
    X = df.drop(columns=["Family"])
    y = df["Family"]

    run_lgbm_pipeline(X, y, config, evaluate_flag=EVALUATE_MODEL)
