import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
import mpld3
import matplotlib.lines as mlines
import json


def run_shap_explainer(model_path, X_train_path, X_test_path, y_test_path, model_name,
                        beeswarm=False, barplot=False, heatmap=False, waterfall=False, decision=False,
                        label_classes_path=None):
    """
    Spustí SHAP explainer na interpretáciu modelu.

    Args:
        model_path (str): Cesta k uloženému modelu (.joblib)
        X_train_path (str): Cesta k trénovacím dátam (CSV)
        X_test_path (str): Cesta k testovacím dátam (CSV)
        y_test_path (str): Cesta k testovacím labelom (CSV)
        model_name (str): Názov modelu (použije sa pre názov výstupného priečinka)
        beeswarm (bool): Generovať beeswarm ploty? Default=False.
        barplot (bool): Generovať bar ploty? Default=False.
        heatmap (bool): Generovať heatmap ploty? Default=False.
        waterfall (bool): Generovať waterfall ploty? Default=False.
        decision (bool): Generovať decision ploty? Default=False.
        label_classes_path (str): Cesta k súboru s uloženými triedami labelov (iba pre XGBoost, nepovinné)
    """

    for path in [model_path, X_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f"❌ Súbor '{path}' neexistuje!")
            return

    print("📥 Načítavam model a dáta...")
    model = load(model_path)

    # Podpora label encoder tried pre XGBoost
    if label_classes_path and os.path.exists(label_classes_path):
        with open(label_classes_path, "r") as f:
            class_names = json.load(f)
    else:
        class_names = model.classes_

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    print("✅ Všetky dátové množiny načítané úspešne.")

    global_vis_dir = f"{model_name}_SHAP_globalne_vizualizacie"
    local_vis_dir = f"{model_name}_SHAP_lokalne_vizualizacie"

    os.makedirs(global_vis_dir, exist_ok=True)
    os.makedirs(os.path.join(global_vis_dir, "vsetky"), exist_ok=True)
    os.makedirs(local_vis_dir, exist_ok=True)

    print(f"📂 Priečinky '{global_vis_dir}' a '{local_vis_dir}' vytvorené.")

    print("⚡ Generujem SHAP hodnoty...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer(X_test)

    if beeswarm:
        print("Vytváram Beeswarm Plot pre všetky triedy zvlášť...")

        num_classes = shap_values_all.values.shape[2]

        plt.rcParams['figure.dpi'] = 200

        # 🔍 5️⃣ Vykreslenie Beeswarm Plot pre každú triedu
        for i in range(num_classes):
            plt.figure(figsize=(35, 20))
            shap.plots.beeswarm(shap_values_all[..., i], max_display=20, show=False,)
            plt.title(f"SHAP Beeswarm Plot - Trieda {class_names[i]}", fontsize=25)
            plt.xlabel(f"SHAP hodnota (vplyv na výstup modelu) - Trieda: {class_names[i]}", fontsize=25)
            plt.ylabel("Atribúty", fontsize=25)
            plt.yticks(rotation=0, fontsize=20)
            plt.subplots_adjust(left=0.4)
            plt.gcf().set_size_inches(35, 20)
            for collection in plt.gca().collections:
                offsets = collection.get_offsets()
                collection.set_sizes([80] * len(offsets))
            plt.tight_layout()
            plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_beeswarm_{class_names[i]}.png"))
            plt.close()

        print("✅ SHAP Beeswarm Ploty pre všetky triedy úspešne uložené!")

    if barplot:
        print("Vytváram Bar Plot pre všetky triedy zvlášť...")

        # Počet tried
        num_classes = shap_values_all.values.shape[2]

        # Nastavenie DPI pre vysokú kvalitu grafov na MacOS
        plt.rcParams['figure.dpi'] = 200  # 🟢 Vyššia kvalita grafov

        # Vytvorenie Beeswarm Plot pre každú triedu
        for i in range(num_classes):
            plt.figure(figsize=(35, 20))  # 🟢 Extra široký a vysoký graf
            shap.plots.bar(shap_values_all[..., i], max_display=20, show=False)
            plt.title(f"SHAP Bar Plot - Trieda {class_names[i]}", fontsize=25)
            plt.xlabel(f"SHAP value (impact on model output) - Rodina: {class_names[i]}", fontsize=25)
            plt.ylabel("Atribúty", fontsize=25)
            plt.yticks(rotation=0, fontsize=20)
            plt.subplots_adjust(left=0.4)  # 🟢 Viac miesta pre názvy atribútov
            plt.gcf().set_size_inches(35, 20)  # 🟢 Full-screen mód pre MacOS

            # 🟢 Zväčšenie guľôčok manuálne pomocou plt.scatter
            for collection in plt.gca().collections:
                offsets = collection.get_offsets()
                collection.set_sizes([80] * len(offsets))  # 🟢 Zväčšenie guľôčok na veľkosť 50

            plt.tight_layout()
            plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_bar_{class_names[i]}.png"))
            plt.close()

        print("SHAP Bar Ploty pre všetky triedy uložené.")

    if heatmap:
        print("Vytváram Heatmap Plot pre všetky triedy zvlášť...")

        # Počet tried
        num_classes = shap_values_all.values.shape[2]

        # Nastavenie DPI pre vysokú kvalitu grafov na MacOS
        plt.rcParams['figure.dpi'] = 200  # 🟢 Vyššia kvalita grafov

        # Vytvorenie Heatmap Plot pre každú triedu
        # Vytvorenie Heatmap Plot pre každú triedu
        for i in range(num_classes):
            plt.figure(figsize=(35, 20))

            instance_order = np.argsort(-np.abs(shap_values_all[..., i].values).sum(1))

            shap.plots.heatmap(
                shap_values_all[..., i],
                max_display=20,
                show=False,
                instance_order=instance_order
            )

            plt.title(f"SHAP Heatmap Plot - Trieda {class_names[i]}", fontsize=25)
            plt.xlabel(f"SHAP value (impact on model output) - Rodina: {class_names[i]}", fontsize=25)
            plt.ylabel("Atribúty", fontsize=25)
            plt.yticks(rotation=0, fontsize=20)
            plt.subplots_adjust(left=0.4)
            plt.gcf().set_size_inches(35, 20)
            plt.tight_layout()
            plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_heatmap_{class_names[i]}.png"))
            plt.close()


        print("SHAP Heatmap Ploty pre všetky triedy uložené.")

    if waterfall:
        print("Vytváram Waterfall Plot pre 5 vzoriek z každej triedy...")

        # Vytvorenie priečinka pre waterfall vizualizácie
        waterfall_vis_dir = os.path.join(local_vis_dir, "waterfall")
        os.makedirs(waterfall_vis_dir, exist_ok=True)
        # Počet vzoriek na vizualizáciu pre každú triedu
        num_local_samples = 5


        # Prejdeme každú triedu a vyberieme 5 vzoriek
        for class_index, class_name in enumerate(class_names):
            class_indices = np.where(np.argmax(shap_values_all.values, axis=2) == class_index)[0][:num_local_samples]
            
            for j, sample_idx in enumerate(class_indices):
                plt.figure(figsize=(35, 20))
                shap.plots.waterfall(shap_values_all[sample_idx, :, class_index], show=False, max_display=20)  # 🟢 Vyberie len SHAP hodnoty pre danú triedu
                plt.title(f"SHAP Waterfall Plot - Trieda {class_name} (vzorka {j+1})")
                plt.subplots_adjust(left=0.4)  # 🟢 Viac miesta pre názvy atribútov
                plt.gcf().set_size_inches(35, 20)  # 🟢 Full-screen mód pre MacOS
                plt.tight_layout()
                plt.savefig(os.path.join(waterfall_vis_dir, f"shap_waterfall_{class_name}_sample{j+1}.png"))
                plt.close()

        print("Waterfall Ploty pre všetky triedy uložené.")

    if decision:
        print("✅ Generujem Multioutput Decision Plot...")

        decision_vis_dir = os.path.join(local_vis_dir, "decision")
        os.makedirs(decision_vis_dir, exist_ok=True)

        # ✅ Vyberieme num_samples_to_plot náhodných vzoriek na vizualizáciu
        num_samples_to_plot = 20
        selected_samples = np.random.choice(X_test.shape[0], num_samples_to_plot, replace=False)

        # ✅ Očakávané hodnoty (Base Values)
        expected_values = explainer.expected_value
        if isinstance(expected_values, np.ndarray):
            expected_values = expected_values.tolist()

        # ✅ SHAP hodnoty pre vybrané vzorky
        shap_values_selected = shap_values_all.values[selected_samples]

        # ✅ Skutočné triedy z `y_test`
        y_test_selected = y_test.iloc[selected_samples, 0].values  

        # ✅ Predikované názvy tried (už sú v stringovej forme)
        y_pred_labels = model.predict(X_test.iloc[selected_samples])

        # ✅ Unikátne triedy v poradí, ako ich model naučil
        unique_classes = model.classes_

        # ✅ Funkcia na generovanie popisov tried do legendy
        def class_labels():
            return [f"{unique_classes[i]}" for i in range(len(unique_classes))]

        # ✅ Transformácia SHAP hodnôt do listu (1 matica pre každú triedu)
        shap_values_list = [shap_values_selected[:, :, i] for i in range(len(unique_classes))]

        # ✅ Generovanie decision plotov pre vybrané vzorky
        for idx, row_index in enumerate(selected_samples):
            actual_class = y_test_selected[idx]  # Skutočná trieda
            predicted_class = y_pred_labels[idx]  # Modelom predikovaná trieda

            # ✅ Overenie správnosti predikcie
            correct_prediction = actual_class == predicted_class
            classification_status = "SPRÁVNA" if correct_prediction else "NESPRÁVNA"

            # ✅ Generovanie decision plotu
            plt.figure(figsize=(30, 15))
            shap.multioutput_decision_plot(
                expected_values,               # Očakávané hodnoty
                shap_values_list,              # SHAP hodnoty ako list
                row_index=idx,                 # Index vzorky
                feature_names=X_test.columns.tolist(),  # Názvy atribútov
                highlight=[np.where(unique_classes == predicted_class)[0][0]],  # Správne indexovanie
                legend_labels=class_labels(),  # Generované popisy tried
                legend_location="lower right",
                show=False
            )

            # ✅ Pridanie textu do grafu
            plt.title(f"SHAP Multioutput Decision Plot - Vzorka {row_index}\n"
                    f"Predikcia: {predicted_class} | Skutočná trieda: {actual_class} "
                    f"({classification_status})", fontsize=14)

            # ✅ Uloženie grafu
            filename = f"shap_multioutput_decision_plot_sample_{row_index}_{classification_status}.png"
            plt.subplots_adjust(left=0.4)  # Viac miesta pre názvy atribútov
            plt.gcf().set_size_inches(30, 15)
            plt.tight_layout()
            plt.savefig(os.path.join(decision_vis_dir, filename))
            plt.close()

        print("✅ Všetky Multioutput Decision Ploty úspešne vygenerované a uložené!")


    print("🎉 Všetky vizualizácie dokončené!")
