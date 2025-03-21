import os
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from joblib import load
import mpld3
import matplotlib.lines as mlines

def run_shap_explainer(model_path, X_train_path, X_test_path, y_test_path, model_name,
                        beeswarm=False, barplot=False, heatmap=False, waterfall=False, decision=False):
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
    """

    # Skontroluj existenciu súborov
    for path in [model_path, X_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f"❌ Súbor '{path}' neexistuje!")
            return

    # Načítanie modelu a dát
    print("📥 Načítavam model a dáta...")
    model = load(model_path)
    class_names = model.classes_

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    print("✅ Všetky dátové množiny načítané úspešne.")

    # Vytvorenie priečinkov pre vizualizácie
    global_vis_dir = f"{model_name}_SHAP_globalne_vizualizacie"
    local_vis_dir = f"{model_name}_SHAP_lokalne_vizualizacie"

    os.makedirs(global_vis_dir, exist_ok=True)
    os.makedirs(os.path.join(global_vis_dir, "vsetky"), exist_ok=True)
    os.makedirs(local_vis_dir, exist_ok=True)

    print(f"📂 Priečinky '{global_vis_dir}' a '{local_vis_dir}' vytvorené.")

    # Výpočet SHAP hodnôt
    print("⚡ Generujem SHAP hodnoty...")
    explainer = shap.TreeExplainer(model)
    shap_values_all = explainer(X_test)

    # -------------------- [ BEESWARM PLOT ] --------------------
    if beeswarm:
        print("📊 Generujem Beeswarm ploty...")
        num_classes = shap_values_all.values.shape[2]
        for i in range(num_classes):
            plt.figure(figsize=(35, 20))
            shap.plots.beeswarm(shap_values_all[..., i], max_display=20, show=False)
            plt.title(f"SHAP Beeswarm Plot - Trieda {class_names[i]}")
            plt.subplots_adjust(left=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_beeswarm_{class_names[i]}.png"))
            plt.close()
        print("✅ Beeswarm ploty uložené!")

    # -------------------- [ BAR PLOT ] --------------------
    if barplot:
        print("📊 Generujem Bar ploty...")
        num_classes = shap_values_all.values.shape[2]
        for i in range(num_classes):
            plt.figure(figsize=(35, 20))
            shap.plots.bar(shap_values_all[..., i], max_display=20, show=False)
            plt.title(f"SHAP Bar Plot - Trieda {class_names[i]}")
            plt.subplots_adjust(left=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_bar_{class_names[i]}.png"))
            plt.close()
        print("✅ Bar ploty uložené!")

    # -------------------- [ HEATMAP PLOT ] --------------------
    if heatmap:
        print("📊 Generujem Heatmap ploty...")
        num_classes = shap_values_all.values.shape[2]
        for i in range(num_classes):
            plt.figure(figsize=(35, 20))
            shap.plots.heatmap(shap_values_all[..., i], max_display=20, show=False)
            plt.title(f"SHAP Heatmap Plot - Trieda {class_names[i]}")
            plt.subplots_adjust(left=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_heatmap_{class_names[i]}.png"))
            plt.close()
        print("✅ Heatmap ploty uložené!")

    # -------------------- [ WATERFALL PLOT ] --------------------
    if waterfall:
        print("📊 Generujem Waterfall ploty...")
        waterfall_vis_dir = os.path.join(local_vis_dir, "waterfall")
        os.makedirs(waterfall_vis_dir, exist_ok=True)

        num_local_samples = 5
        for j in range(num_local_samples):
            plt.figure(figsize=(35, 20))
            shap.plots.waterfall(shap_values_all[j, :, 0], show=False)
            plt.title(f"SHAP Waterfall Plot - Vzorka {j+1}")
            plt.subplots_adjust(left=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(waterfall_vis_dir, f"shap_waterfall_sample{j+1}.png"))
            plt.close()
        print("✅ Waterfall ploty uložené!")

    # -------------------- [ DECISION PLOT ] --------------------
    if decision:
        print("📊 Generujem Decision plot...")
        decision_vis_dir = os.path.join(local_vis_dir, "decision")
        os.makedirs(decision_vis_dir, exist_ok=True)

        num_samples_to_plot = 20
        selected_samples = np.random.choice(X_test.shape[0], num_samples_to_plot, replace=False)

        expected_values = explainer.expected_value
        shap_values_list = [shap_values_all.values[selected_samples, :, i] for i in range(len(class_names))]

        for idx, row_index in enumerate(selected_samples):
            plt.figure(figsize=(30, 15))
            shap.multioutput_decision_plot(
                expected_values,
                shap_values_list,
                row_index=idx,
                feature_names=X_test.columns.tolist(),
                legend_labels=[f"{class_names[i]}" for i in range(len(class_names))],
                legend_location="lower right",
                show=False
            )

            plt.title(f"SHAP Decision Plot - Vzorka {row_index}")
            plt.subplots_adjust(left=0.4)
            plt.tight_layout()
            plt.savefig(os.path.join(decision_vis_dir, f"shap_decision_plot_sample_{row_index}.png"))
            plt.close()

        print("✅ Decision ploty uložené!")

    print("🎉 Všetky vizualizácie dokončené!")
