import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
import mpld3
import matplotlib.lines as mlines
import json
import pathlib


def run_shap_explainer(model_path, X_train_path, X_test_path, y_test_path, model_name,
                        beeswarm=False, barplot=False, heatmap=False, waterfall=False, decision=False,
                        label_classes_path=None):

    for path in [model_path, X_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f" Súbor '{path}' neexistuje!")
            return

    print("Načítavam model a dáta...")
    model = load(model_path)

    # Podpora label encoder tried pre XGBoost
    # Získanie názvov tried
    if label_classes_path and os.path.exists(label_classes_path):
        ext = pathlib.Path(label_classes_path).suffix.lower()
        if ext == ".json":
            with open(label_classes_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        elif ext == ".npy":
            class_names = np.load(label_classes_path, allow_pickle=True).tolist()
        else:
            print(f" Nepodporovaný formát súboru: {ext}")
            return
        print("Triedy načítané z label_classes súboru.")
    else:
        class_names = model.classes_


    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    global_vis_dir = f"{model_name}_SHAP_globalne_vizualizacie"
    local_vis_dir = f"{model_name}_SHAP_lokalne_vizualizacie"

    os.makedirs(global_vis_dir, exist_ok=True)
    os.makedirs(os.path.join(global_vis_dir, "All"), exist_ok=True)
    os.makedirs(local_vis_dir, exist_ok=True)

    print(f" Priečinky '{global_vis_dir}' a '{local_vis_dir}' vytvorené.")

    print("⚡ Generujem SHAP hodnoty...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer(X_test)

    if beeswarm:
        print("Vytváram Beeswarm Plot pre všetky triedy zvlášť...")

        num_classes = shap_values_all.values.shape[2]

        plt.rcParams['figure.dpi'] = 200

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
            plt.savefig(os.path.join(global_vis_dir, "All", f"shap_beeswarm_{class_names[i]}.png"))
            plt.close()

        print("SHAP Beeswarm Ploty pre všetky triedy boli úspešne uložené")

    if barplot:
        print("Vytváram Bar Plot pre všetky triedy zvlášť...")

        num_classes = shap_values_all.values.shape[2]

        plt.rcParams['figure.dpi'] = 200

        for i in range(num_classes):
            plt.figure(figsize=(35, 20))
            shap.plots.bar(shap_values_all[..., i], max_display=20, show=False)
            plt.title(f"SHAP Bar Plot - Trieda {class_names[i]}", fontsize=25)
            plt.xlabel(f"SHAP value (impact on model output) - Rodina: {class_names[i]}", fontsize=25)
            plt.ylabel("Atribúty", fontsize=25)
            plt.yticks(rotation=0, fontsize=20)
            plt.subplots_adjust(left=0.4)
            plt.gcf().set_size_inches(35, 20)

            for collection in plt.gca().collections:
                offsets = collection.get_offsets()
                collection.set_sizes([80] * len(offsets))

            plt.tight_layout()
            plt.savefig(os.path.join(global_vis_dir, "All", f"shap_bar_{class_names[i]}.png"))
            plt.close()

        print("SHAP Bar Ploty pre všetky triedy boli uložené.")

    if heatmap:
        print("Vytváram Heatmap Plot pre všetky triedy zvlášť...")

        num_classes = shap_values_all.values.shape[2]

        plt.rcParams['figure.dpi'] = 200

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
            plt.savefig(os.path.join(global_vis_dir, "All", f"shap_heatmap_{class_names[i]}.png"))
            plt.close()


        print("SHAP Heatmap Ploty pre všetky triedy boli uložené.")



    if waterfall:
        print("Vytváram Waterfall Plot pre náhodné vzorky...")

        waterfall_vis_dir = os.path.join(local_vis_dir, "waterfall")
        os.makedirs(waterfall_vis_dir, exist_ok=True)

        num_samples = 70
        selected_samples = np.random.choice(X_test.shape[0], num_samples, replace=False)

        for j, sample_idx in enumerate(selected_samples):
            sample_input = X_test.iloc[[sample_idx]]

            predicted_label = model.predict(sample_input)[0]

            if isinstance(predicted_label, (int, np.integer)):
                predicted_class_name = class_names[predicted_label]
                class_index = predicted_label
            else:
                predicted_class_name = predicted_label
                try:
                    class_index = list(model.classes_).index(predicted_class_name)
                except ValueError:
                    print(f" Trieda {predicted_class_name} nebola nájdená v model.classes_: {model.classes_}")
                    continue

            actual_class_raw = y_test.iloc[sample_idx, 0]
            if actual_class_raw in class_names:
                actual_class_name = actual_class_raw
            elif isinstance(actual_class_raw, (int, np.integer)) and actual_class_raw < len(class_names):
                actual_class_name = class_names[actual_class_raw]
            else:
                print(f" Nepodarilo sa získať názov triedy pre y_test hodnotu: {actual_class_raw}")
                continue

            correct_prediction = predicted_class_name == actual_class_name
            classification_status = "SPRÁVNA" if correct_prediction else "NESPRÁVNA"

            plt.figure(figsize=(35, 20))
            shap.plots.waterfall(shap_values_all[sample_idx, :, class_index], show=False, max_display=20)

            plt.title(
                f"SHAP Waterfall Plot - Vzorka {sample_idx}\n"
                f"Predikcia: {predicted_class_name} | Skutočná trieda: {actual_class_name} ({classification_status})"
            )

            plt.subplots_adjust(left=0.4)
            plt.gcf().set_size_inches(35, 20)
            plt.tight_layout()

            filename = f"shap_waterfall_sample_{sample_idx}_{classification_status}.png"
            plt.savefig(os.path.join(waterfall_vis_dir, filename))
            plt.close()

        print("Waterfall Ploty pre náhodné vzorky boli úspešne uložené")

    if decision:
        print("Generujem Multioutput Decision Plot...")

        decision_vis_dir = os.path.join(local_vis_dir, "decision")
        os.makedirs(decision_vis_dir, exist_ok=True)

        num_samples_to_plot = 20
        selected_samples = np.random.choice(X_test.shape[0], num_samples_to_plot, replace=False)

        expected_values = explainer.expected_value
        if isinstance(expected_values, np.ndarray):
            expected_values = expected_values.tolist()

        shap_values_selected = shap_values_all.values[selected_samples]

        y_test_selected = y_test.iloc[selected_samples, 0].values  

        y_pred_labels = model.predict(X_test.iloc[selected_samples])

        unique_classes = model.classes_

        def class_labels():
            return [f"{class_names[i]}" for i in range(len(class_names))]

        shap_values_list = [shap_values_selected[:, :, i] for i in range(len(unique_classes))]

        for idx, row_index in enumerate(selected_samples):
            actual_class = y_test_selected[idx] 
            predicted_class = y_pred_labels[idx]
            
            correct_prediction = actual_class == predicted_class
            classification_status = "SPRÁVNA" if correct_prediction else "NESPRÁVNA"

            plt.figure(figsize=(30, 15))
            shap.multioutput_decision_plot(
                expected_values,
                shap_values_list,
                row_index=idx,
                feature_names=X_test.columns.tolist(),
                highlight=[np.where(unique_classes == predicted_class)[0][0]],
                legend_labels=class_labels(),
                legend_location="lower right",
                show=False,
            )

            plt.title(f"SHAP Multioutput Decision Plot - Vzorka {row_index}\n"
                    f"Predikcia: {predicted_class} | Skutočná trieda: {actual_class} "
                    f"({classification_status})", fontsize=14)

            filename = f"shap_multioutput_decision_plot_sample_{row_index}_{classification_status}.png"
            plt.subplots_adjust(left=0.4)
            plt.gcf().set_size_inches(30, 15)
            plt.tight_layout()
            plt.savefig(os.path.join(decision_vis_dir, filename))
            plt.close()

        print("Všetky Multioutput Decision Ploty boli úspešne vygenerované a uložené")


    print("Všetky vizualizácie boli dokončené")
