from InquirerPy import inquirer

def main():
    while True:
        # Hlavné menu
        main_choice = inquirer.select(
            message="Vyber si akciu:",
            choices=[
                "Trenovanie klasifikacneho modelu",
                "Vysvetlitelnost modelu",
                "Exit"
            ],
            default=None,
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
                        "Exit"
                    ],
                    default=None,
                ).execute()

                if model_choice == "Exit":
                    break
                print(f"\nZvolil si model: {model_choice}\n")
                # Tu sa neskôr pridá logika tréningu

        elif main_choice == "Vysvetlitelnost modelu":
            while True:
                explain_choice = inquirer.select(
                    message="Vyber si metodu vysvetlitelnosti:",
                    choices=[
                        "SHAP (Global & Local Explainer)",
                        "LIME (Local Explainer)",
                        "Exit"
                    ],
                    default=None,
                ).execute()

                if explain_choice == "Exit":
                    break
                print(f"\nZvolil si vysvetlitelnost: {explain_choice}\n")
                # Tu sa neskôr pridá logika vysvetlenia

        elif main_choice == "Exit":
            print("\nUkončujem aplikáciu. Maj sa!\n")
            break

if __name__ == "__main__":
    main()
