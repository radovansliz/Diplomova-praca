import pandas as pd
import numpy as np
import shap
import sys
import os
import matplotlib.pyplot as plt
from joblib import load
import mpld3  # Importujeme mpld3 pre HTML expo

# Skontroluj, či sú zadané všetky argumenty
if len(sys.argv) != 5:
    print("Použitie: python shap_analysis.py <cesta_k_modelu.joblib> <cesta_k_X_train.csv> <cesta_k_X_test.csv> <nazov_modelu>")
    sys.exit(1)

# Načítanie argumentov
model_path = sys.argv[1]
data_path = sys.argv[2]
data_path2 = sys.argv[3]
model_name = sys.argv[4].lower()


# Skontroluj existenciu súborov
if not os.path.exists(model_path):
    print(f"Súbor modelu '{model_path}' neexistuje!")
    sys.exit(1)

if not os.path.exists(data_path):
    print(f"Súbor dát '{data_path}' neexistuje!")
    sys.exit(1)

if not os.path.exists(data_path2):
    print(f"Súbor dát '{data_path2}' neexistuje!")
    sys.exit(1)

# Načítanie modelu a dát
print("Načítavam model a dáta...")
model = load(model_path)
X_train = pd.read_csv(data_path)
X_test = pd.read_csv(data_path2)


# Skontroluj, či majú rovnaké stĺpce
if list(X_train.columns) != list(X_test.columns):
    print("X_train a X_test nemajú rovnaké stĺpce!")
    sys.exit(1)

# # Ručne zadané názvy tried
nazvy_tried = ["Airpush", "Inoco", "Locker", "Mytrackp", "Shedun", 
               "skymobi", "Slocker", "Smforw", "smsagent", "smsreg", 
               "smsthief", "Styricka"]

# Generovanie SHAP hodnôt
print("Generujem SHAP hodnoty...")
explainer = shap.TreeExplainer(model)  # funguje pre stromové modely, pre iné treba zmeniť
shap_values = explainer.shap_values(X_test)

# # Vizuálna interpretácia pomocou SHAP
# shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
# plt.savefig("shap_summary_plot.png")
# print("SHAP summary plot uložený ako shap_summary_plot.png")

# # Export do HTML pomocou mpld3
# html_str = mpld3.fig_to_html(plt.gcf())
# with open("shap_summary_plot.html", "w") as f:
#     f.write(html_str)
# print("SHAP summary plot uložený ako shap_summary_plot.html")




# Vypočítanie priemerných SHAP hodnôt pre každý atribút
shap_mean = np.mean(np.abs(shap_values), axis=0)
important_features = np.argsort(shap_mean)[-20:][::-1]  # Vyber top 20 najdôležitejších atribútov

print(f"Important features: {important_features}")

# Získanie názvov atribútov z X_test
feature_names = X_test.columns

# Preklad indexov na názvy atribútov
important_feature_names = [[feature_names[idx] for idx in row] for row in important_features]

# Zobrazenie výsledkov
for i, features in enumerate(important_feature_names):
    print(f"Trieda {i + 1}: {features}")


print("Tvar shap_values:", np.array(shap_values).shape)
print("Tvar shap_values[0]:", shap_values[0].shape)
print("Tvar X_test:", X_test.shape)
print("Tvar X_train:", X_train.shape)


# Rozdelenie shap_values na 12 matíc (po jednej pre každú triedu)
shap_values_per_class = [shap_values[:, :, i] for i in range(12)]
print(f"Shap values per class: {shap_values_per_class}")




# ----------------- PRVY GRAF -----------------
import matplotlib.pyplot as plt

# Názvy tried podľa tvojho zoznamu
nazvy_tried = ["Airpush", "Inoco", "Locker", "Mytrackp", "Shedun", 
               "skymobi", "Slocker", "Smforw", "smsagent", "smsreg", 
               "smsthief", "Styricka"]

# Generovanie SHAP plotov pre každú triedu s názvom
for class_idx in range(12):
    print(f"Vytváram SHAP summary plot pre triedu {nazvy_tried[class_idx]}...")
    
    # Získanie názvov najdôležitejších atribútov pre konkrétnu triedu
    top_features = X_test.columns[important_features[class_idx]]
    
    # Generovanie SHAP Summary Plot pre danú triedu
    shap.summary_plot(shap_values_per_class[class_idx][:, important_features[class_idx]], 
                      X_test[top_features], 
                      show=False)
    
    # Uloženie grafu ako PNG s názvom triedy
    plt.title(f"SHAP Summary Plot pre {nazvy_tried[class_idx]}")
    plt.savefig(f"shap_summary_{nazvy_tried[class_idx]}.png")
    plt.close()

print("✅ Všetky SHAP grafy boli úspešne vytvorené a uložené s názvami tried!")









# # Vyber náhodnú vzorku z tréningových dát pre SHAP analýzu
# X_shap = X_train.sample(100, random_state=42)

# # Výber správneho explainera podľa typu modelu
# print("Vyberám správny SHAP explainer podľa typu modelu...")

# if model_name in ["randomforest", "extratrees", "xgboost", "lgbm", "lightgbm", "bagging"]:
#     explainer = shap.TreeExplainer(model)
#     print("Použitý SHAP explainer: TreeExplainer")
# elif model_name == "adaboost":
#     explainer = shap.TreeExplainer(model, X_shap, model_output="margin")
#     print("Použitý SHAP explainer: TreeExplainer pre AdaBoost")
# else:
#     print("Nepodporovaný typ modelu pre tento skript.")
#     sys.exit(1)


# # Ručne zadané názvy tried
# nazvy_tried = ["Airpush", "Inoco", "Locker", "Mytrackp", "Shedun", 
#                "skymobi", "Slocker", "Smforw", "smsagent", "smsreg", 
#                "smsthief", "Styricka"]

# # Výpočet SHAP hodnôt
# print("Počítam SHAP hodnoty...")
# shap_values = explainer.shap_values(X_shap)


# print(f"Počet vlastností v X_shap: {X_shap.shape[1]}")
# print(f"Počet vlastností v shap_values: {shap_values[0].shape[1] if isinstance(shap_values, list) else shap_values.shape[1]}")
# print("Názvy stĺpcov v X_shap:", X_shap.columns.tolist()[:5], "...")  # Iba prvých 5 pre kontrolu
# print(f"SHAP VALUES SHAPE: {shap_values.shape}")



# # Vytvor priečinok pre uloženie výsledkov
# output_dir = "shap_results_v2"
# os.makedirs(output_dir, exist_ok=True)

# # Uloženie SHAP hodnôt
# np.save(os.path.join(output_dir, "shap_values.npy"), shap_values)
# X_shap.to_csv(os.path.join(output_dir, "X_shap_sample.csv"), index=False)


# ------------- LOKALNA WATERFALL INTERPRETACIA ---------------

# Výber indexu pre vzor (môžeš zmeniť podľa potreby)
# index = 0  # Prvý vzor

# # Pre každý class_index a názov triedy v nazvy_tried
# for class_index, nazov_triedy in enumerate(nazvy_tried):
#     print(f"Generujem waterfall graf pre vzor {index} a triedu {nazov_triedy} (index {class_index})...")

#     # Vytvorenie waterfall grafu pre vybraný vzor a triedu
#     shap.initjs()
#     fig, ax = plt.subplots(figsize=(10, 6))  # Vytvor nový obrázok a os

#     shap.waterfall_plot(shap.Explanation(values=shap_values[index][:, class_index], 
#                                          base_values=explainer.expected_value[class_index], 
#                                          data=X_shap.iloc[index],
#                                          feature_names=X_shap.columns), show=False)

#     # Uloženie grafu do súboru s názvom podľa triedy
#     waterfall_path = os.path.join(output_dir, f"waterfall_plot_{nazov_triedy}_sample_{index}.png")
#     plt.savefig(waterfall_path, dpi=300, bbox_inches='tight')
#     plt.close()  # Zavrie obrázok, aby sa neprekrývali grafy pri viacerých generovaniach

#     print(f"Waterfall graf uložený do súboru: {waterfall_path}")


# -------- GLOBALNE INTERPRETACIE ---------

# Vytvor priečinok pre uloženie globálnych grafov, ak ešte neexistuje
# global_dir = os.path.join(output_dir, "global_interpretation")
# os.makedirs(global_dir, exist_ok=True)

# # Výpis tvaru shap_values
# print(f"Tvar shap_values: {len(shap_values)} tried.")
# if isinstance(shap_values, list):
#     for i, sv in enumerate(shap_values):
#         print(f"Tvar shap_values pre triedu {i}: {sv.shape}")
# else:
#     print(f"Pre tento model je shap_values numpy pole s tvarom: {shap_values.shape}")

# # Výpis tvaru X_shap
# print(f"Tvar X_shap: {X_shap.shape}")

# # Výpis počtu tried v modeli (pre viac tried)
# try:
#     # Ak používame model, ktorý poskytuje počet tried, môžeme získať počet tried z modelu
#     # Ak je to napríklad scikit-learn model, môžeme sa pozrieť na model.classes_
#     print(f"Počet tried v modeli: {model.n_classes_ if hasattr(model, 'n_classes_') else 'Nepodporované'}")
# except Exception as e:
#     print(f"Chyba pri získavaní počtu tried: {str(e)}")


# print("Generujem Summary Plot (Bee Swarm Plot)...")

# # Vytvorenie summary plotu pre všetky triedy
# shap.initjs()
# fig, ax = plt.subplots(figsize=(12, 8))  # Nastavenie veľkosti grafu

# shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns, plot_type="bee_swarm", show=False)
# plt.legend(nazvy_tried, title="Triedy")

# # Uloženie grafu do súboru
# summary_plot_path = os.path.join(global_dir, "summary_plot.png")
# plt.legend(nazvy_tried, title="Triedy")
# plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
# plt.close()  # Zavrie graf, aby sa neprekrývali pri ďalších generovaniach

# print(f"Summary Plot uložený do súboru: {summary_plot_path}")


# # TOP 20
# print("Generujem Summary Plot pre top 20 vlastností...")

# fig, ax = plt.subplots(figsize=(10, 8))
# shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns, max_display=20, show=False)
# plt.legend(nazvy_tried, title="Triedy")

# summary_top20_plot_path = os.path.join(global_dir, "summary_top20_plot.png")
# plt.savefig(summary_top20_plot_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Summary Plot pre top 20 vlastností uložený do súboru: {summary_top20_plot_path}")

# # Logaritmicka skalovatelnosti
# print("Generujem Summary Plot s logaritmickou osou...")

# fig, ax = plt.subplots(figsize=(10, 8))
# shap.summary_plot(shap_values, X_shap, feature_names=X_shap.columns, use_log_scale=True, show=False)
# plt.legend(nazvy_tried, title="Triedy")

# summary_log_plot_path = os.path.join(global_dir, "summary_log_plot.png")
# plt.savefig(summary_log_plot_path, dpi=300, bbox_inches='tight')
# plt.close()

# print(f"Summary Plot s logaritmickou osou uložený do súboru: {summary_log_plot_path}")

# Výpis tvaru shap_values pre jednu triedu (napr. pre triedu s indexom 0)
# print(f"Tvar shap_values pre triedu 0: {shap_values[0].shape}")

# # Ukážka hodnôt shap_values pre jednu triedu (prvých 5 riadkov pre triedu 0)
# print("Ukážka hodnot SHAP pre triedu 0 (prvých 5 riadkov):")
# print(shap_values[0][:5])

# # Výpis tvaru X_shap a ukážka prvých riadkov
# print(f"Tvar X_shap: {X_shap.shape}")
# print("Ukážka prvých riadkov X_shap:")
# print(X_shap.head())  # Zobrazí prvých 5 riadkov

# # Výpis prvých 5 názvov vlastností v X_shap
# print("Prvé 5 názvov vlastností v X_shap:")
# print(X_shap.columns.tolist()[:5])

# # Kontrola hodnoty base_value pre triedu 0
# print(f"Base value pre triedu 0: {explainer.expected_value[0]}")


# Generovanie Summary Plot pre každú triedu osobitne
# for class_index, nazov_triedy in enumerate(nazvy_tried):
#     print(f"Generujem Summary Plot pre triedu {nazov_triedy} (index {class_index})...")

#     # Výber správnych SHAP hodnôt pre danú triedu (2D pole)
#     shap_values_class = shap_values[:, :, class_index]  # Vyberieme hodnoty len pre jednu triedu

#     # Skontroluj, či má shap_values_class správny tvar
#     if shap_values_class.shape[0] != X_shap.shape[0] or shap_values_class.shape[1] != X_shap.shape[1]:
#         print(f"Chyba: Tvar shap_values_class pre triedu {nazov_triedy} sa nezhoduje s tvarom X_shap.")
#         continue  # Preskočíme triedu, ak sa tvar nezhoduje

#     # Vytvorenie summary plotu pre danú triedu
#     fig, ax = plt.subplots(figsize=(12, 8))  # Nastavenie veľkosti grafu
#     shap.summary_plot(shap_values_class, X_shap, feature_names=X_shap.columns, plot_type="bee_swarm", show=False)

#     # Nastavíme názvy osí a prispôsobíme graf
#     ax.set_xlabel("SHAP interaction value")  # Nastavenie popisu pre os X
#     ax.set_ylabel("Features")  # Nastavenie popisu pre os Y

#     # Uloženie grafu pre každú triedu
#     summary_class_plot_path = os.path.join(output_dir, f"summary_plot_{nazov_triedy}.png")
#     plt.savefig(summary_class_plot_path, dpi=300, bbox_inches='tight')
#     plt.close()

#     print(f"Summary Plot pre triedu {nazov_triedy} uložený do súboru: {summary_class_plot_path}")




# Vytvorenie Summary Plotu pre agregované SHAP hodnoty (pre všetky triedy)
shap.initjs()

# # Vytvoriť novú matplotlib figúru
# fig, ax = plt.subplots(figsize=(12, 8))

# # Vytvorenie summary plotu pre všetky vzorky
# shap.summary_plot(
#     shap_values, 
#     X_shap, 
#     feature_names=X_shap.columns, 
#     plot_type="bee_swarm", 
#     show=False  # Nepotrebujeme zobrazovať graf hneď
# )

# # Uložíme Summary Plot do súboru
# summary_plot_path = os.path.join(output_dir, "summary_plot_mean.png")
# plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
# plt.close()  # Zavrieme graf po uložení

# print(f"Summary Plot uložený do súboru: {summary_plot_path}")












# print("Generujem farebný SHAP summary plot pre všetky triedy...")
# plt.figure(figsize=(20, 10))
# shap.summary_plot(shap_values, X_shap, class_names=nazvy_tried, show=False)
# plt.legend(nazvy_tried, title="Triedy")
# plt.savefig(os.path.join(output_dir, "shap_summary_plot_all_classes.png"), bbox_inches='tight')
# plt.close()



# # Summary plot – celkový prehľad dôležitosti vlastností (bar plot)
# print("Generujem a ukladám SHAP summary plot (bar)...")
# plt.figure()
# shap.summary_plot(shap_values, X_shap, plot_type="bar", show=False)
# plt.savefig(os.path.join(output_dir, "shap_summary_plot_bar.png"))
# plt.close()

# print("Generujem SHAP summary plot...")
# plt.figure(figsize=(16, 10))
# shap_values_mean = np.mean(shap_values, axis=2)
# shap.summary_plot(shap_values_mean, X_shap, show=False)
# plt.savefig(os.path.join(output_dir, "shap_summary_plot_mean.png"), bbox_inches='tight')
# plt.close()


# # Detailný summary plot
# print("Generujem a ukladám detailný SHAP summary plot...")
# plt.figure()
# shap.summary_plot(shap_values, X_shap, show=False)
# plt.savefig(os.path.join(output_dir, "shap_summary_plot.png"))
# plt.close()

# # # Force plot – vysvetlenie konkrétnej predikcie
# # print("Generujem a ukladám SHAP force plot pre prvý záznam...")
# # force_plot = shap.force_plot(explainer.expected_value[1], shap_values[1], X_shap.iloc[1, :].to_numpy(), feature_names=X_shap.columns)
# # shap.save_html(os.path.join(output_dir, "shap_force_plot.html"), force_plot)





# print("Generujem a ukladám interaktívny SHAP decision plot ako HTML...")
# for trieda in range(len(nazvy_tried)):
#     plt.figure(figsize=(20, 10))
#     decision_plot = shap.decision_plot(
#         explainer.expected_value[trieda],
#         shap_values[:, :, trieda],
#         X_shap,
#         feature_names=list(X_shap.columns),
#         show=False,
#         title=f"SHAP Decision Plot pre {nazvy_tried[trieda]}"
#     )
    
#     # Uloženie PNG grafu
#     plt.savefig(os.path.join(output_dir, f"shap_decision_plot_{nazvy_tried[trieda]}.png"), bbox_inches='tight')
    
#     # # Uloženie ako interaktívne HTML pomocou `mpld3`
#     # html_str = mpld3.fig_to_html(plt.gcf())  # Prevod aktuálneho grafu na HTML
#     # with open(os.path.join(output_dir, f"shap_decision_plot_{nazvy_tried[trieda]}.html"), "w") as html_file:
#     #     html_file.write(html_str)
    
#     plt.close()


# print("Generujem SHAP beeswarm plot...")
# plt.figure(figsize=(20, 10))
# shap.summary_plot(shap_values_mean, X_shap, plot_type="dot", show=False)
# plt.savefig(os.path.join(output_dir, "shap_beeswarm_plot.png"), bbox_inches='tight')
# plt.close()


# print("Generujem SHAP force plot pre konkrétnu vzorku a triedu...")
# for trieda in range(len(nazvy_tried)):
#     force_plot = shap.force_plot(
#         explainer.expected_value[trieda],
#         shap_values[0, :, trieda],
#         X_shap.iloc[0, :].to_numpy(),
#         feature_names=X_shap.columns
#     )
#     shap.save_html(os.path.join(output_dir, f"shap_force_plot_{nazvy_tried[trieda]}.html"), force_plot)


# print("Generujem SHAP dependence plot pre najdôležitejšiu vlastnosť...")
# plt.figure(figsize=(20, 10))
# shap.dependence_plot("Memory_PssTotal", shap_values_mean, X_shap, show=False)
# plt.savefig(os.path.join(output_dir, "shap_dependence_plot_Memory_PssTotal.png"), bbox_inches='tight')
# plt.close()



# print("Generujem SHAP heatmap plot...")
# plt.figure(figsize=(20, 10))

# # Priemerovanie cez všetky triedy
# shap_values_mean = np.mean(shap_values, axis=2)

# # Výpočet korelačnej matice z priemerných absolútnych hodnôt
# shap_heatmap = np.corrcoef(shap_values_mean.T)

# # Kontrola tvaru pre istotu
# print("Tvar interakčnej matice:", shap_heatmap.shape)

# plt.imshow(shap_heatmap, cmap="viridis", aspect="auto")
# plt.colorbar()
# plt.title("SHAP Heatmap (Korelačná matica)")
# plt.savefig(os.path.join(output_dir, "shap_heatmap_plot.png"), bbox_inches='tight')
# plt.close()


# # print("Generujem SHAP bar plot pre TOP 20 dôležitých vlastností...")

# # print("Skracujem názvy vlastností pre SHAP bar plot...")
# # print("Minimálna hodnota SHAP:", np.min(shap_values))
# # print("Maximálna hodnota SHAP:", np.max(shap_values))
# # print("Priemerná absolútna hodnota SHAP:", np.mean(np.abs(shap_values)))

# # print("Normalizujem SHAP hodnoty pre lepšiu viditeľnosť...")
# # shap_values_mean = np.mean(np.abs(shap_values), axis=2)
# # shap_max = np.max(shap_values_mean)
# # shap_values_normalized = shap_values_mean / shap_max  # Normalizácia na rozsah 0–1

# # plt.figure(figsize=(30, 20))
# # shap.summary_plot(shap_values_normalized, X_shap, plot_type="barh", max_display=30, show=False)
# # plt.savefig(os.path.join(output_dir, "shap_bar_plot_normalized.png"), bbox_inches='tight')
# # plt.close()


# # max_len = 30
# # short_columns = [col if len(col) <= max_len else col[:max_len] + "..." for col in X_shap.columns]
# # X_shap.columns = short_columns  # Nahradenie dlhých názvov skrátenými
# # plt.figure(figsize=(20, 20))

# # # Výpočet priemerných absolútnych hodnôt
# # shap_importance = np.mean(np.abs(shap_values_mean), axis=0)

# # # Získanie indexov TOP 10 vlastností
# # top_indices = np.argsort(shap_importance)[-10:]

# # # Vytvorenie nového DataFrame len s TOP 10 vlastnosťami
# # X_shap_top = X_shap.iloc[:, top_indices]
# # shap_values_mean_top = shap_values_mean[:, top_indices]

# # # Generovanie grafu
# # shap.summary_plot(shap_values_mean_top, X_shap_top, plot_type="barh", show=False)
# # plt.savefig(os.path.join(output_dir, "shap_bar_plot_top10.png"), bbox_inches='tight')
# # plt.close()




# # Summary Plot: Pre rýchly prehľad dôležitosti vlastností.
# # Beeswarm Plot: Na pochopenie distribúcie vplyvov medzi vzorkami.
# # Force Plot: Na vysvetlenie konkrétnych predikcií.
# # Dependence Plot: Na analýzu interakcií medzi vlastnosťami.
# # Interaction Summary Plot: Na identifikáciu kľúčových interakcií.
# # Heatmap Plot: Na vizualizáciu vzťahov medzi vlastnosťami.
# # Bar Plot: Na zoradenie vlastností podľa dôležitosti.





# print(f"Všetky SHAP výstupy boli uložené do priečinka '{output_dir}'.")
 