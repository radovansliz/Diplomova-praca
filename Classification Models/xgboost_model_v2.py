import os
import sys
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, learning_curve
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# Funkcia na nastavenie logovania do súboru a konzoly
def setup_logger(log_file):
    logger = logging.getLogger('xgboost_logger')
    logger.setLevel(logging.DEBUG)
    # Vyčistenie predchádzajúcich handlerov, ak existujú
    if logger.hasHandlers():
        logger.handlers.clear()
    # Vytvorenie file handler-a
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.DEBUG)
    # Vytvorenie stream handler-a pre konzolu
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.DEBUG)
    # Nastavenie formátovania
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

# Funkcia na vykreslenie a uloženie konfúznej matice
def plot_confusion_matrix(cm, classes, fold, output_dir='plots'):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap=plt.cm.Blues,
                xticklabels=classes, yticklabels=classes)
    plt.ylabel('Skutočné triedy')
    plt.xlabel('Predikované triedy')
    plt.title(f'Konfúzna matica - Fold {fold}')
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'confusion_matrix_fold_{fold}.png')
    plt.savefig(plot_path)
    plt.close()

# Funkcia na vykreslenie a uloženie learning curve
def plot_learning_curve(estimator, X, y, fold, output_dir='plots', cv=5, n_jobs=-1):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv,
                                                            n_jobs=n_jobs,
                                                            train_sizes=np.linspace(0.1, 1.0, 5))
    train_scores_mean = np.mean(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    
    plt.figure()
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Tréningová skóre")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="CV skóre")
    plt.title(f'Learning Curve - Fold {fold}')
    plt.xlabel("Počet trénovacích príkladov")
    plt.ylabel("Skóre")
    plt.legend(loc="best")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f'learning_curve_fold_{fold}.png')
    plt.savefig(plot_path)
    plt.close()

def main():
    # Nastavenie logovacieho súboru
    log_file = 'training_log.txt'
    logger = setup_logger(log_file)
    logger.info("Spúšťam klasifikáciu malwaru pomocou XGBoost a 10-násobnej krížovej validácie.")

    # Načítanie datasetu (predpokladáme súbor dataset.csv)
    try:
        data = pd.read_csv('12k_samples_12_families.csv')
        logger.info("Dataset bol úspešne načítaný.")
    except Exception as e:
        logger.error(f"Chyba pri načítaní datasetu: {e}")
        return

    # Odstránenie stĺpcov 'Hash' a 'Category'
    for col in ['Hash', 'Category']:
        if col in data.columns:
            data.drop(columns=[col], inplace=True)
            logger.info(f"Stĺpec '{col}' bol odstránený.")
        else:
            logger.warning(f"Stĺpec '{col}' sa nenašiel v datasete.")

    # Kontrola, či je prítomný stĺpec 'Family' (cieľová trieda)
    if 'Family' not in data.columns:
        logger.error("Stĺpec 'Family' (cieľová trieda) sa nenašiel v datasete.")
        return

    # Rozdelenie dát na vstupné atribúty X a cieľovú premennú y
    X = data.drop(columns=['Family'])
    y = data['Family']

    # Kódovanie cieľových tried (ak sú nečíselné)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    logger.info("Cieľové triedy boli zakódované pomocou LabelEncoder.")

    # Definovanie 10-násobnej stratifikovanej krížovej validácie
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    fold_no = 1
    accuracies = []

    # Krížová validácia
    for train_index, test_index in skf.split(X, y_encoded):
        logger.info(f"Spúšťam fold {fold_no}")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        # Inicializácia modelu XGBoost
        model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        
        # Tréning modelu s evaluačným setom pre zobrazenie priebehu
        eval_set = [(X_test, y_test)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=True)
        logger.info(f"Tréning modelu dokončený pre fold {fold_no}.")

        # Predikcia a vyhodnotenie
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)
        logger.info(f"Fold {fold_no} - Presnosť: {acc:.4f}")
        
        # Vytvorenie a uloženie konfúznej matice
        cm = confusion_matrix(y_test, y_pred)
        logger.info(f"Fold {fold_no} - Konfúzna matica:\n{cm}")
        plot_confusion_matrix(cm, classes=le.classes_, fold=fold_no)

        # Vytvorenie a uloženie learning curve na trénovacej množine
        plot_learning_curve(model, X_train, y_train, fold=fold_no)

        # Uloženie podrobného reportu klasifikácie
        report = classification_report(y_test, y_pred)
        logger.info(f"Fold {fold_no} - Report klasifikácie:\n{report}")

        fold_no += 1

    # Výpočet a zaznamenanie priemernej presnosti
    avg_accuracy = np.mean(accuracies)
    logger.info(f"Priemerná presnosť cez 10 foldov: {avg_accuracy:.4f}")

if __name__ == '__main__':
    main()
