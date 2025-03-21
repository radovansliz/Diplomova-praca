import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3  # Na interaktívne HTML vizualizácie
from joblib import load
from lime.lime_tabular import LimeTabularExplainer

def run_lime_explainer(model_path, X_train_path, X_test_path, y_test_path, model_name, n_samples=5):
    """
    Spustí LIME Local Explainer na interpretáciu modelu.

    Args:
        model_path (str): Cesta k uloženému modelu (.joblib)
        X_train_path (str): Cesta k trénovacím dátam (CSV)
        X_test_path (str): Cesta k testovacím dátam (CSV)
        y_test_path (str): Cesta k testovacím labelom (CSV)
        model_name (str): Názov modelu (použije sa pre názov výstupného priečinka)
        n_samples (int, optional): Počet vzoriek na vysvetlenie. Default = 5.
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
    y_test = pd.read_csv(y_test_path).values.flatten()

    print("✅ Všetky dátové množiny načítané úspešne.")

    # Vytvorenie priečinka na vizualizácie
    local_vis_dir = f"{model_name}_LIME_lokalne_vizualizacie"
    os.makedirs(local_vis_dir, exist_ok=True)
    print(f"📂 Priečinok '{local_vis_dir}' vytvorený.")

    # Inicializácia LIME explaineru
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True
    )

    # Generovanie vysvetlení pre n_samples náhodných vzoriek
    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        sample = X_test.iloc[idx].values.reshape(1, -1)
        explanation = explainer.explain_instance(
            sample[0], model.predict_proba, num_features=20, top_labels=len(class_names)
        )

        # Uloženie HTML vizualizácie
        html_path = os.path.join(local_vis_dir, f"lime_explanation_{i}.html")
        explanation.save_to_file(html_path)
        print(f"✅ LIME vysvetlenie uložené: {html_path}")

        # Uloženie obrázkov atribútov pre každú triedu
        for label in range(len(class_names)):
            plt.figure(figsize=(35, 20))
            fig = explanation.as_pyplot_figure(label=label)
            plt.title(f"LIME vysvetlenie pre vzorku {i} - Trieda: {class_names[label]}")
            plt.subplots_adjust(left=0.4)
            plt.gcf().set_size_inches(35, 20)
            plt.tight_layout()
            plt.savefig(os.path.join(local_vis_dir, f"lime_explanation_{i}_class_{label}.png"))
            plt.close()

    print("🎉 Všetky vysvetlenia boli úspešne vygenerované.")
