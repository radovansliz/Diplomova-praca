import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mpld3
import json
from joblib import load
from lime.lime_tabular import LimeTabularExplainer
import pathlib


def run_lime_explainer(model_path, X_train_path, X_test_path, y_test_path, model_name, n_samples=5, label_classes_path=None):
    for path in [model_path, X_train_path, X_test_path, y_test_path]:
        if not os.path.exists(path):
            print(f"Súbor '{path}' neexistuje!")
            return

    print("📥 Načítavam model a dáta...")
    model = load(model_path)

    if label_classes_path and os.path.exists(label_classes_path):
        ext = pathlib.Path(label_classes_path).suffix.lower()
        if ext == ".json":
            with open(label_classes_path, "r", encoding="utf-8") as f:
                class_names = json.load(f)
        elif ext == ".npy":
            class_names = np.load(label_classes_path, allow_pickle=True).tolist()
        else:
            print(f"Nepodporovaný formát súboru: {ext}")
            return
        print("Triedy načítané z label_classes súboru.")
    else:
        class_names = model.classes_

    X_train = pd.read_csv(X_train_path)
    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.flatten()

    print("Všetky dátové množiny načítané úspešne.")

    local_vis_dir = f"{model_name}_LIME_lokalne_vizualizacie"
    os.makedirs(local_vis_dir, exist_ok=True)
    print(f"Priečinok '{local_vis_dir}' vytvorený.")

    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=class_names,
        mode="classification",
        discretize_continuous=True
    )

    sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

    for i, idx in enumerate(sample_indices):
        sample = X_test.iloc[[idx]]  # zachová názvy stĺpcov
        explanation = explainer.explain_instance(
            sample.values[0], model.predict_proba, num_features=20, top_labels=len(class_names)
        )

        predicted_label = model.predict(sample)[0]
        true_label = y_test[idx]

        # Rozhodni, či sú labely čísla alebo stringy
        if isinstance(predicted_label, (int, np.integer)) and isinstance(true_label, (int, np.integer)):
            pred_name = class_names[predicted_label]
            true_name = class_names[true_label]
        else:
            pred_name = predicted_label
            true_name = true_label

        correct = pred_name == true_name
        correctness = "CORRECT" if correct else "WRONG"

        print(f"Vzorka {i} - Skutočný: {true_name}, Predikovaný: {pred_name} - {'SPRÁVNE' if correct else 'NESPRÁVNE'}")

        html_filename = f"lime_explanation_{i}_REAL_{true_name}_PRED_{pred_name}_{correctness}.html"
        html_path = os.path.join(local_vis_dir, html_filename)
        explanation.save_to_file(html_path, labels=explanation.available_labels())
        print(f"LIME vysvetlenie uložené: {html_path}")

        for label in range(len(class_names)):
            plt.figure(figsize=(35, 20))
            fig = explanation.as_pyplot_figure(label=label)
            plt.title(f"LIME vysvetlenie pre vzorku {i} - Trieda: {class_names[label]}")
            plt.subplots_adjust(left=0.4)
            plt.gcf().set_size_inches(35, 20)
            plt.tight_layout()

            png_filename = f"lime_explanation_{i}_class_{label}_REAL_{true_name}_PRED_{pred_name}_{correctness}.png"
            png_path = os.path.join(local_vis_dir, png_filename)
            plt.savefig(png_path)
            plt.close()

    print("Všetky vysvetlenia boli úspešne vygenerované.")
