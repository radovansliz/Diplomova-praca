import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import mpld3  # Na interaktívne HTML vizualizácie
from joblib import load
from lime.lime_tabular import LimeTabularExplainer

# -------------------------- [ KONTROLA ARGUMENTOV ] ----------------------
if len(sys.argv) != 6:
    print("Použitie: python lime_local_explainer.py <cesta_k_modelu.joblib> <cesta_k_X_train.csv> <cesta_k_X_test.csv> <cesta_k_y_test.csv> <nazov_modelu>")
    sys.exit(1)

model_path = sys.argv[1]
data_path = sys.argv[2]
data_path2 = sys.argv[3]
data_path3 = sys.argv[4]
model_name = sys.argv[5].lower()

# Skontroluj existenciu súborov
for path in [model_path, data_path, data_path2, data_path3]:
    if not os.path.exists(path):
        print(f"Súbor '{path}' neexistuje!")
        sys.exit(1)

# -------------------------- [ NAČÍTANIE DÁT ] ----------------------
print("Načítavam model a dáta...")
model = load(model_path)
class_names = model.classes_

X_train = pd.read_csv(data_path)
X_test = pd.read_csv(data_path2)
y_test = pd.read_csv(data_path3).values.flatten()

print("Všetky dátové množiny načítané úspešne.")

# -------------------------- [ VYTVORENIE PRIEČINKOV ] ----------------------
local_vis_dir = f"{model_name}_LIME_lokalne_vizualizacie"
os.makedirs(local_vis_dir, exist_ok=True)
print("Priečinky pre vizualizácie vytvorené úspešne.")

# -------------------------- [ LIME EXPLAINER ] ----------------------
explainer = LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns.tolist(),
    class_names=class_names,
    mode="classification",
    discretize_continuous=True
)

# -------------------------- [ GENEROVANIE VIZUALIZÁCIÍ ] ----------------------
n_samples = 5  # Počet vzoriek na vysvetlenie
sample_indices = np.random.choice(len(X_test), n_samples, replace=False)

for i, idx in enumerate(sample_indices):
    sample = X_test.iloc[idx].values.reshape(1, -1)
    explanation = explainer.explain_instance(sample[0], model.predict_proba, num_features=20, top_labels=len(class_names))
    
    # Uloženie HTML vizualizácie
    html_path = os.path.join(local_vis_dir, f"lime_explanation_{i}.html")
    explanation.save_to_file(html_path)
    print(f"LIME vysvetlenie uložené: {html_path}")
    
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

print("Všetky vysvetlenia boli úspešne vygenerované.")
