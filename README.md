# Diplomova-praca

Diplomová práca - Vysvetliteľná Detekcia Malware

Používané modely na klasifikáciu v rámci práce:

- Random Forest
- Extra Trees Classifier (ETC)
- Bagging Classifier (BC)
- XGBoost (XGB)
- AdaBoost (ABC)
- Light Gradient Boosting Machine (LGBM)

Používané metódy vysvetliteľnosti v rámci práce:

- SHAP
- LIME

# Konzolová aplikácia – Návod na použitie

**Aktuálna verzia konzolovej aplikácie sa nachádza v priečinku `Final Solution`.**

## Požiadavky

- Aktuálna verzia **Python 3**
- Inštalácia knižníc pomocou správcu balíkov `pip`

### Inštalácia závislostí

```bash
pip install -r requirements.txt
```

### Spustenie aplikácie

```bash
python3 main_console.py
```

---

## Funkcionalita aplikácie

Po spustení aplikácie `main_console.py` sa používateľovi zobrazí interaktívne CLI menu s dvoma hlavnými možnosťami:

- _Tréning klasifikačného modelu_
- _Vysvetlenie modelu (XAI metódy)_

---

### Tréning klasifikačného modelu

Používateľ má možnosť vybrať si jeden z podporovaných modelov:

- AdaBoost Classifier
- Bagging Classifier
- ExtraTrees Classifier
- LGBM Classifier
- Random Forest Classifier
- XGBoost Classifier

#### Požadované vstupy:

- Cesta k CSV súboru s dátami (optimalizované pre dataset použitý v práci)
- Cesta k JSON súboru s parametrami modelu (dostupné v priečinku `Configs`)
- Potvrdenie, či sa má spustiť evaluácia modelu po trénovaní

#### Výstupy tréningu:

- `model.joblib` – uložený model
- `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv`
- `xgboost_label_classes.npy` – len pri XGBoost modeli
- (voliteľne pri zapnutej evaluácii):
  - `classification report`
  - `confusion matrix`
  - `learning curve`

---

### Vysvetlenie modelu (XAI metódy)

Používateľ má možnosť zvoliť jednu z dvoch vysvetliteľných metód:

- **SHAP (Global & Local Explainer)**
- **LIME (Local Explainer)**

#### Vstupy pre obe metódy:

- `model.joblib` – natrénovaný klasifikátor
- `X_train.csv`, `X_test.csv` – vstupné atribúty
- `y_test.csv` – cieľové hodnoty pre testovaciu množinu
- `model_name` – používateľský názov modelu (použije sa pre výstupné priečinky)
- `xgboost_label_classes.npy` – súbor pre XGBoost model na zobrazovanie označení klasifikovaných tried

---

### SHAP

Aplikácia interaktívne ponúkne výber typov vizualizácií, ktoré sa majú vygenerovať:

- Beeswarm
- Barplot
- Heatmap
- Waterfall
- Decision

#### Výstupné priečinky:

- `<model_name>_SHAP_globalne_vizualizacie/`
- `<model_name>_SHAP_lokalne_vizualizacie/`

---

### LIME

Používateľ zadá počet vzoriek, ktoré sa majú analyzovať. Pre každú vzorku sa vytvoria:

- HTML vizualizácia vysvetlenia (`.html`)
- Grafické zobrazenie pre každú triedu (`.png`)

#### Výstupný priečinok:

- `<model_name>_LIME_lokalne_vizualizacie/`
