from InquirerPy import inquirer
import os
import json
import pandas as pd

from Models.XGBoost_Classification_Model import run_xgboost_pipeline
from Models.Adaboost_Classification_Model import run_adaboost_pipeline
from Models.Bagging_Classifier_Model import run_bagging_pipeline
from Models.Extra_Trees_Classification_Model import run_etc_pipeline
from Models.LGBM_Classifier_Model import run_lgbm_pipeline
from Models.Random_Forest_Classification_Model import run_rf_pipeline

from XAI_Models.SHAP_XAI import run_shap_explainer
from XAI_Models.LIME_XAI import run_lime_explainer

MODEL_FUNCTIONS = {
    "XGBoost Classifier": run_xgboost_pipeline,
    "AdaBoost Classifier": run_adaboost_pipeline,
    "Bagging Classifier": run_bagging_pipeline,
    "ExtraTrees Classifier": run_etc_pipeline,
    "LGBM Classifier": run_lgbm_pipeline,
    "Random Forest Classifier": run_rf_pipeline,
}

def train_model(csv_path, json_path, model_type, evaluation_model):
    # Načítanie konfigurácie
    try:
        with open(json_path, "r") as config_file:
            config = json.load(config_file)
    except Exception as e:
        print(f"Chyba pri načítaní JSON konfigurácie: {e}")
        return

    # Načítanie datasetu
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"Chyba pri načítaní CSV súboru: {e}")
        return

    # Definovanie vstupných a výstupných hodnôt
    try:
        X = df.drop(columns=["Family", "Hash", "Category"])
        y = df["Family"]
    except KeyError as e:
        print(f"Chýbajúce stĺpce v datasete: {e}")
        return

    print(f"\nDataset načítaný: {df.shape[0]} riadkov, {df.shape[1]} stĺpcov")
    print(f"Model: {model_type}")
    print(f"Evaluácia bude spustená: {evaluation_model}")
    print(f"Konfigurácia: {config}")

    if model_type in MODEL_FUNCTIONS:
        print(f"Spúšťam tréning modelu {model_type}...")
        MODEL_FUNCTIONS[model_type](X, y, config, evaluate_flag=evaluation_model)
        print(f"Tréning modelu {model_type} dokončený!\n")
    else:
        print(f"Model '{model_type}' nie je podporovaný.")

def explain_model(explainer_type):
    model_type = inquirer.select(
        message="Vyber klasifikátor, ktorý bol použitý na trénovanie:",
        choices=[
            "XGBoost Classifier",
            "AdaBoost Classifier",
            "Bagging Classifier",
            "ExtraTrees Classifier",
            "LGBM Classifier",
            "Random Forest Classifier"
        ]).execute()

    model_path = inquirer.text(message="Zadaj cestu k modelu (.joblib):").execute()
    X_train_path = inquirer.text(message="Zadaj cestu k trénovacím dátam (.csv):").execute()
    X_test_path = inquirer.text(message="Zadaj cestu k testovacím dátam (.csv):").execute()
    y_test_path = inquirer.text(message="Zadaj cestu k testovacím labelom (.csv):").execute()
    model_name = inquirer.text(message="Zadaj názov modelu (bude použitý v názvoch priečinkov):").execute()

    label_classes_path = None
    if model_type == "XGBoost Classifier":
        label_classes_path = inquirer.text(
            message="Zadaj cestu k súboru s label classes (xgboost_label_classes.npy): [nepovinné]"
        ).execute()
        if label_classes_path.strip() == "":
            label_classes_path = None

    for path in [model_path, X_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f"Súbor '{path}' neexistuje!")
            return

    if explainer_type == "SHAP (Global & Local Explainer)":
        shap_visualizations = inquirer.checkbox(
            message="Vyber SHAP vizualizácie, ktoré chceš vygenerovať:",
            choices=["Beeswarm", "Barplot", "Heatmap", "Waterfall", "Decision"]
        ).execute()

        run_shap_explainer(
            model_path=model_path,
            X_train_path=X_train_path,
            X_test_path=X_test_path,
            y_test_path=y_test_path,
            model_name=model_name,
            beeswarm="Beeswarm" in shap_visualizations,
            barplot="Barplot" in shap_visualizations,
            heatmap="Heatmap" in shap_visualizations,
            waterfall="Waterfall" in shap_visualizations,
            decision="Decision" in shap_visualizations,
            label_classes_path=label_classes_path
        )

    elif explainer_type == "LIME (Local Explainer)":
        n_samples = int(inquirer.text(
            message="Koľko vzoriek chceš vysvetliť? (default: 5)"
        ).execute() or 5)

        run_lime_explainer(
            model_path=model_path,
            X_train_path=X_train_path,
            X_test_path=X_test_path,
            y_test_path=y_test_path,
            model_name=model_name,
            n_samples=n_samples,
            label_classes_path=label_classes_path
        )

def main():
    while True:
        main_choice = inquirer.select(
            message="Vyber si akciu:",
            choices=[
                "Trenovanie klasifikacneho modelu",
                "Vysvetlitelnost modelu",
                "Ukončiť"
            ],
        ).execute()

        if main_choice == "Trenovanie klasifikacneho modelu":
            while True:
                model_choice = inquirer.select(
                    message="Vyber si klasifikacny model na trenovanie:",
                    choices=[
                        "AdaBoost Classifier",
                        "Bagging Classifier",
                        "ExtraTrees Classifier",
                        "LGBM Classifier",
                        "Random Forest Classifier",
                        "XGBoost Classifier",
                        "Späť"
                    ],
                ).execute()

                if model_choice == "Späť":
                    break

                csv_path = inquirer.text(message="Zadaj cestu k CSV súboru s dátami:").execute()
                json_path = inquirer.text(message="Zadaj cestu k JSON súboru s konfiguráciou:").execute()

                evaluation_model = inquirer.confirm(
                    message="Chceš po trénovaní spustiť evaluáciu a uložiť výsledky?",
                    default=True
                ).execute()

                if not os.path.exists(csv_path):
                    print(f"Súbor {csv_path} neexistuje.")
                    continue
                if not os.path.exists(json_path):
                    print(f"Súbor {json_path} neexistuje.")
                    continue

                train_model(csv_path, json_path, model_choice, evaluation_model)

        elif main_choice == "Vysvetlitelnost modelu":
            while True:
                explain_choice = inquirer.select(
                    message="Vyber si metodu vysvetlitelnosti:",
                    choices=[
                        "SHAP (Global & Local Explainer)",
                        "LIME (Local Explainer)",
                        "Späť"
                    ],
                ).execute()

                if explain_choice == "Späť":
                    break

                explain_model(explain_choice)

        elif main_choice == "Ukončiť":
            print("\nUkončujem aplikáciu. Maj sa!\n")
            break

if __name__ == "__main__":
    main()
