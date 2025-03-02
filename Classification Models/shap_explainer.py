import pandas as pd
import numpy as np
import shap
import sys
import os
import matplotlib.pyplot as plt
from joblib import load
import mpld3  # Importujeme mpld3 pre HTML expo

# Skontroluj, či sú zadané všetky argumenty
if len(sys.argv) != 4:
    print("Použitie: python shap_analysis.py <cesta_k_modelu.joblib> <cesta_k_X_train.csv> <nazov_modelu>")
    sys.exit(1)

# Načítanie argumentov
model_path = sys.argv[1]
data_path = sys.argv[2]
model_name = sys.argv[3].lower()

# Skontroluj existenciu súborov
if not os.path.exists(model_path):
    print(f"Súbor modelu '{model_path}' neexistuje!")
    sys.exit(1)

if not os.path.exists(data_path):
    print(f"Súbor dát '{data_path}' neexistuje!")
    sys.exit(1)

# Načítanie modelu a dát
print("Načítavam model a dáta...")
model = load(model_path)
X_train = pd.read_csv(data_path)

# Vyber náhodnú vzorku z tréningových dát pre SHAP analýzu
X_shap = X_train.sample(100, random_state=42)

# Výber správneho explainera podľa typu modelu
print("Vyberám správny SHAP explainer podľa typu modelu...")

if model_name in ["randomforest", "extratrees", "xgboost", "lgbm", "lightgbm", "bagging"]:
    explainer = shap.TreeExplainer(model)
    print("Použitý SHAP explainer: TreeExplainer")
elif model_name == "adaboost":
    explainer = shap.TreeExplainer(model, X_shap, model_output="margin")
    print("Použitý SHAP explainer: TreeExplainer pre AdaBoost")
else:
    print("Nepodporovaný typ modelu pre tento skript.")
    sys.exit(1)



# Výpočet SHAP hodnôt
print("Počítam SHAP hodnoty...")
shap_values = explainer.shap_values(X_shap)


print(f"Počet vlastností v X_shap: {X_shap.shape[1]}")
print(f"Počet vlastností v shap_values: {shap_values[0].shape[1] if isinstance(shap_values, list) else shap_values.shape[1]}")
print("Názvy stĺpcov v X_shap:", X_shap.columns.tolist()[:5], "...")  # Iba prvých 5 pre kontrolu


# Vytvor priečinok pre uloženie výsledkov
output_dir = "shap_results"
os.makedirs(output_dir, exist_ok=True)

# Uloženie SHAP hodnôt
np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
X_shap.to_csv(os.path.join(output_dir, "X_shap_sample.csv"), index=False)

# Ručne zadané názvy tried
nazvy_tried = ["Airpush", "Inoco", "Locker", "Mytrackp", "Shedun", 
               "skymobi", "Slocker", "Smforw", "smsagent", "smsreg", 
               "smsthief", "Styricka"]

print("Názvy tried:", nazvy_tried)

print("Generujem farebný SHAP summary plot pre všetky triedy...")
plt.figure(figsize=(20, 10))
shap.summary_plot(shap_values, X_shap, class_names=nazvy_tried, show=False)
plt.legend(nazvy_tried, title="Triedy")
plt.savefig(os.path.join(output_dir, "shap_summary_plot_all_classes.png"), bbox_inches='tight')
plt.close()



# Summary plot – celkový prehľad dôležitosti vlastností (bar plot)
print("Generujem a ukladám SHAP summary plot (bar)...")
plt.figure()
shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
plt.savefig(os.path.join(output_dir, "shap_summary_plot_bar.png"))
plt.close()

print("Generujem SHAP summary plot...")
plt.figure(figsize=(16, 10))
shap_values_mean = np.mean(shap_values, axis=2)
shap.summary_plot(shap_values_mean, X_shap, show=False)
plt.savefig(os.path.join(output_dir, "shap_summary_plot_mean.png"), bbox_inches='tight')
plt.close()


# Detailný summary plot
print("Generujem a ukladám detailný SHAP summary plot...")
plt.figure()
shap.summary_plot(shap_values, X_shap, show=False)
plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
plt.close()

# # Force plot – vysvetlenie konkrétnej predikcie
# print("Generujem a ukladám SHAP force plot pre prvý záznam...")
# force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], X_shap.iloc[1, :].to_numpy(), feature_names=X_shap.columns)
# shap.save_html(os.path.join(output_dir, "shap_force_plot.html"), force_plot)





print("Generujem a ukladám interaktívny SHAP decision plot ako HTML...")
for trieda in range(len(nazvy_tried)):
    plt.figure(figsize=(20, 10))
    decision_plot = shap.decision_plot(
        explainer.expected_value[trieda],
        shap_values[:, :, trieda],
        X_shap,
        feature_names=list(X_shap.columns),
        show=False,
        title=f"SHAP Decision Plot pre {nazvy_tried[trieda]}"
    )
    
    # Uloženie PNG grafu
    plt.savefig(os.path.join(output_dir, f"shap_decision_plot_{nazvy_tried[trieda]}.png"), bbox_inches='tight')
    
    # # Uloženie ako interaktívne HTML pomocou `mpld3`
    # html_str = mpld3.fig_to_html(plt.gcf())  # Prevod aktuálneho grafu na HTML
    # with open(os.path.join(output_dir, f"shap_decision_plot_{nazvy_tried[trieda]}.html"), "w") as html_file:
    #     html_file.write(html_str)
    
    plt.close()


print("Generujem SHAP beeswarm plot...")
plt.figure(figsize=(20, 10))
shap.summary_plot(shap_values_mean, X_shap, plot_type="dot", show=False)
plt.savefig(os.path.join(output_dir, "shap_beeswarm_plot.png"), bbox_inches='tight')
plt.close()


print("Generujem SHAP force plot pre konkrétnu vzorku a triedu...")
for trieda in range(len(nazvy_tried)):
    force_plot = shap.force_plot(
        explainer.expected_value[trieda],
        shap_values[0, :, trieda],
        X_shap.iloc[0, :].to_numpy(),
        feature_names=X_shap.columns
    )
    shap.save_html(os.path.join(output_dir, f"shap_force_plot_{nazvy_tried[trieda]}.html"), force_plot)


print("Generujem SHAP dependence plot pre najdôležitejšiu vlastnosť...")
plt.figure(figsize=(20, 10))
shap.dependence_plot("Memory_PssTotal", shap_values_mean, X_shap, show=False)
plt.savefig(os.path.join(output_dir, "shap_dependence_plot_Memory_PssTotal.png"), bbox_inches='tight')
plt.close()



print("Generujem SHAP heatmap plot...")
plt.figure(figsize=(20, 10))

# Priemerovanie cez všetky triedy
shap_values_mean = np.mean(shap_values, axis=2)

# Výpočet korelačnej matice z priemerných absolútnych hodnôt
shap_heatmap = np.corrcoef(shap_values_mean.T)

# Kontrola tvaru pre istotu
print("Tvar interakčnej matice:", shap_heatmap.shape)

plt.imshow(shap_heatmap, cmap="viridis", aspect="auto")
plt.colorbar()
plt.title("SHAP Heatmap (Korelačná matica)")
plt.savefig(os.path.join(output_dir, "shap_heatmap_plot.png"), bbox_inches='tight')
plt.close()


# print("Generujem SHAP bar plot pre TOP 20 dôležitých vlastností...")

# print("Skracujem názvy vlastností pre SHAP bar plot...")
# print("Minimálna hodnota SHAP:", np.min(shap_values))
# print("Maximálna hodnota SHAP:", np.max(shap_values))
# print("Priemerná absolútna hodnota SHAP:", np.mean(np.abs(shap_values)))

# print("Normalizujem SHAP hodnoty pre lepšiu viditeľnosť...")
# shap_values_mean = np.mean(np.abs(shap_values), axis=2)
# shap_max = np.max(shap_values_mean)
# shap_values_normalized = shap_values_mean / shap_max  # Normalizácia na rozsah 0–1

# plt.figure(figsize=(30, 20))
# shap.summary_plot(shap_values_normalized, X_shap, plot_type="barh", max_display=30, show=False)
# plt.savefig(os.path.join(output_dir, "shap_bar_plot_normalized.png"), bbox_inches='tight')
# plt.close()


# max_len = 30
# short_columns = [col if len(col) <= max_len else col[:max_len] + "..." for col in X_shap.columns]
# X_shap.columns = short_columns  # Nahradenie dlhých názvov skrátenými
# plt.figure(figsize=(20, 20))

# # Výpočet priemerných absolútnych hodnôt
# shap_importance = np.mean(np.abs(shap_values_mean), axis=0)

# # Získanie indexov TOP 10 vlastností
# top_indices = np.argsort(shap_importance)[-10:]

# # Vytvorenie nového DataFrame len s TOP 10 vlastnosťami
# X_shap_top = X_shap.iloc[:, top_indices]
# shap_values_mean_top = shap_values_mean[:, top_indices]

# # Generovanie grafu
# shap.summary_plot(shap_values_mean_top, X_shap_top, plot_type="barh", show=False)
# plt.savefig(os.path.join(output_dir, "shap_bar_plot_top10.png"), bbox_inches='tight')
# plt.close()




# Summary Plot: Pre rýchly prehľad dôležitosti vlastností.
# Beeswarm Plot: Na pochopenie distribúcie vplyvov medzi vzorkami.
# Force Plot: Na vysvetlenie konkrétnych predikcií.
# Dependence Plot: Na analýzu interakcií medzi vlastnosťami.
# Interaction Summary Plot: Na identifikáciu kľúčových interakcií.
# Heatmap Plot: Na vizualizáciu vzťahov medzi vlastnosťami.
# Bar Plot: Na zoradenie vlastností podľa dôležitosti.





print(f"Všetky SHAP výstupy boli uložené do priečinka '{output_dir}'.")
 