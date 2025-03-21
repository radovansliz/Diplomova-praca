from InquirerPy import inquirer
import pandas as pd
import json
import os

def train_model(csv_path, json_path, model_type, evaluation_model):
    import pandas as pd
    import json
    import os

    # Načítanie konfigurácie z JSON súboru
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

    # Vytvorenie vstupov a výstupov
    try:
        X = df.drop(columns=["Family", "Hash", "Category"])
        y = df["Family"]
    except KeyError as e:
        print(f"Chýbajúce stĺpce v datasete: {e}")
        return

    # Debug výpisy
    print(f"\n✅ Dataset načítaný: {df.shape[0]} riadkov, {df.shape[1]} stĺpcov")
    print(f"➡️ Model: {model_type}")
    print(f"🧠 Evaluácia bude spustená: {evaluation_model}")
    print(f"🔧 Konfigurácia: {config}")

    # Tu neskôr implementuj: tréning modelu, evaluáciu, uloženie výsledkov


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

                # Cesta k CSV súboru
                csv_path = inquirer.text(
                    message="Zadaj cestu k CSV súboru s dátami:"
                ).execute()

                # Cesta k JSON konfigurácii
                json_path = inquirer.text(
                    message="Zadaj cestu k JSON súboru s konfiguráciou:"
                ).execute()

                # Chce používateľ spustiť evaluáciu?
                evaluation_model = inquirer.confirm(
                    message="Chceš po trénovaní spustiť evaluáciu a uložiť výsledky?",
                    default=True
                ).execute()

                # Overenie existencie
                if not os.path.exists(csv_path):
                    print(f"Súbor {csv_path} neexistuje.")
                    continue
                if not os.path.exists(json_path):
                    print(f"Súbor {json_path} neexistuje.")
                    continue

                # Načítanie súborov
                try:
                    data = pd.read_csv(csv_path)
                    with open(json_path, "r") as f:
                        config = json.load(f)
                except Exception as e:
                    print(f"Chyba pri načítaní súborov: {e}")
                    continue

                # Zavolanie hlavnej funkcie
                train_model(data, config, model_choice, evaluation_model)

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
                print(f"\nZvolil si vysvetlitelnost: {explain_choice}\n")
                # Tu sa neskôr pridá logika vysvetlenia

        elif main_choice == "Exit":
            print("\nUkončujem aplikáciu. Maj sa!\n")
            break

if __name__ == "__main__":
    main()
