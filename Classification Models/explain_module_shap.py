import pandas as pd
import numpy as np
import shap
import sys
import os
import matplotlib.pyplot as plt
from joblib import load
import mpld3  # Importujeme mpld3 pre HTML export
import matplotlib.lines as mlines  # Na manuálne pridanie legendy

# -------------------------- [ KONTROLA ARGUMENTOV ] ----------------------
if len(sys.argv) != 5:
    print("Použitie: python explain_module_shap.py <cesta_k_modelu.joblib> <cesta_k_X_train.csv> <cesta_k_X_test.csv> <nazov_modelu>")
    sys.exit(1)

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

# -------------------------- [ NAČÍTANIE DÁT ] ----------------------
print("Načítavam model a dáta...")
model = load(model_path)

X_train = pd.read_csv(data_path)
X_test = pd.read_csv(data_path2)

X_train_mixed = pd.read_csv("X_train_mixed.csv")
X_train_trojan = pd.read_csv("X_train_trojan.csv")
X_test_mixed = pd.read_csv("X_test_mixed.csv")
X_test_trojan = pd.read_csv("X_test_trojan.csv")

print("Všetky dátové množiny načítané úspešne.")

# -------------------------- [ VYTVORENIE PRIEČINKOV ] ----------------------
global_vis_dir = "Shap_globalne_vizualizacie"
local_vis_dir = "Shap_lokalne_vizualizacie"

os.makedirs(global_vis_dir, exist_ok=True)
os.makedirs(os.path.join(global_vis_dir, "vsetky"), exist_ok=True)
os.makedirs(os.path.join(global_vis_dir, "mixed_malware"), exist_ok=True)
os.makedirs(os.path.join(global_vis_dir, "trojan_malware"), exist_ok=True)
os.makedirs(local_vis_dir, exist_ok=True)

print("Priečinky pre vizualizácie vytvorené úspešne.")

# -------------------------- [ VZORKOVANIE 100 DÁT PRED VÝPOČTOM SHAP ] ----------------------
X_test_sample = X_test.sample(100, random_state=42)
X_test_mixed_sample = X_test_mixed.sample(100, random_state=42)
X_test_trojan_sample = X_test_trojan.sample(100, random_state=42)

print("Vzorkovanie 100 dát úspešné.")

# -------------------------- [ MAPA KATEGÓRIÍ ] ----------------------
nazvy_tried = ["airpush", "inoco", "iocker", "mytrackp", "Sshedun", 
               "skymobi", "slocker", "smforw", "smsagent", "smsreg", 
               "smsthief", "styricka"]

wider_categories = {
    "Trojan": ["smsthief", "smforw", "mytrackp", "styricka", "smsagent", "locker"],
    "Mixed_Malware": ["shedun", "inoco", "airpush", "smsreg", "skymobi", "slocker"]
}

# -------------------------- [ KONTROLA SPRÁVNEHO ROZDELENIA RODÍN ] ----------------------
# Priradenie rodín do správnych kategórií
trojan_families = ["smsthief", "smforw", "mytrackp", "styricka", "smsagent", "locker"]
mixed_malware_families = ["shedun", "inoco", "airpush", "smsreg", "skymobi", "slocker"]

# Overenie správnosti dát pred výpočtom SHAP hodnôt
print("Kontrolujem rodiny v dátach...")
print("Rodiny v Mixed Malware:", set(X_test_mixed_sample.columns) & set(trojan_families))
print("Rodiny v Trojan:", set(X_test_trojan_sample.columns) & set(mixed_malware_families))

# -------------------------- [ VÝPOČET SHAP HODNÔT IBA PRE KOREKTNÉ DÁTA ] ----------------------
print("Generujem SHAP hodnoty pre všetky skupiny...")
explainer = shap.TreeExplainer(model)

# SHAP hodnoty len pre dané kategórie dát
# shap_values_all = explainer.shap_values(X_test_sample)
# shap_values_mixed = explainer.shap_values(X_test_mixed_sample)
# shap_values_trojan = explainer.shap_values(X_test_trojan_sample)

shap_values_all = explainer(X_test)
shap_values_mixed = explainer(X_test_mixed_sample)
shap_values_trojan = explainer(X_test_trojan_sample)

# -------------------------- [ Nastavenie zapnutia spustenia vizualizacii ] ----------------------

beeswarm = False
barplot = False
heatmap = True  # 🟢 Pridaný prepínač pre heatmap

# -------------------------- [ BEESWARM PLOT PRE VŠETKY TRIEDY ZVLÁŠŤ ] ----------------------
if beeswarm:
    print("Vytváram Beeswarm Plot pre všetky triedy zvlášť...")

    # Počet tried
    num_classes = shap_values_all.values.shape[2]
    class_names = ["airpush", "inoco", "iocker", "mytrackp", "Sshedun", 
                "skymobi", "slocker", "smforw", "smsagent", "smsreg", 
                "smsthief", "styricka"]

    # Nastavenie DPI pre vysokú kvalitu grafov na MacOS
    plt.rcParams['figure.dpi'] = 200  # 🟢 Vyššia kvalita grafov

    # Vytvorenie Beeswarm Plot pre každú triedu
    for i in range(num_classes):
        plt.figure(figsize=(35, 20))  # 🟢 Extra široký a vysoký graf
        shap.plots.beeswarm(shap_values_all[..., i], max_display=20, show=False)
        plt.title(f"SHAP Beeswarm Plot - Trieda {class_names[i]}", fontsize=25)
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
        plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_beeswarm_{class_names[i]}.png"))
        plt.close()

    print("SHAP Beeswarm Ploty pre všetky triedy uložené.")


## -------------------------- [ Bar PLOT PRE VŠETKY TRIEDY ZVLÁŠŤ ] ----------------------
if barplot:
    print("Vytváram Bar Plot pre všetky triedy zvlášť...")

    # Počet tried
    num_classes = shap_values_all.values.shape[2]
    class_names = ["airpush", "inoco", "iocker", "mytrackp", "Sshedun", 
                "skymobi", "slocker", "smforw", "smsagent", "smsreg", 
                "smsthief", "styricka"]

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

# -------------------------- [ HEATMAP PLOT PRE VŠETKY TRIEDY ZVLÁŠŤ ] ----------------------
if heatmap:
    print("Vytváram Heatmap Plot pre všetky triedy zvlášť...")

    # Počet tried
    num_classes = shap_values_all.values.shape[2]
    class_names = ["airpush", "inoco", "iocker", "mytrackp", "Sshedun", 
                   "skymobi", "slocker", "smforw", "smsagent", "smsreg", 
                   "smsthief", "styricka"]

    # Nastavenie DPI pre vysokú kvalitu grafov na MacOS
    plt.rcParams['figure.dpi'] = 200  # 🟢 Vyššia kvalita grafov

    # Vytvorenie Heatmap Plot pre každú triedu
    for i in range(num_classes):
        plt.figure(figsize=(35, 20))  # 🟢 Extra široký a vysoký graf
        shap.plots.heatmap(shap_values_all[..., i], max_display=20, show=False, instance_order=shap_values_all.sum(1))
        plt.title(f"SHAP Heatmap Plot - Trieda {class_names[i]}", fontsize=25)
        plt.xlabel(f"SHAP value (impact on model output) - Rodina: {class_names[i]}", fontsize=25)
        plt.ylabel("Atribúty", fontsize=25)
        plt.yticks(rotation=0, fontsize=20)
        plt.subplots_adjust(left=0.4)  # 🟢 Viac miesta pre názvy atribútov
        plt.gcf().set_size_inches(35, 20)  # 🟢 Full-screen mód pre MacOS

        plt.tight_layout()
        plt.savefig(os.path.join(global_vis_dir, "vsetky", f"shap_heatmap_{class_names[i]}.png"))
        plt.close()

    print("SHAP Heatmap Ploty pre všetky triedy uložené.")


