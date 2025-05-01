import pandas as pd
import os
import glob

def count_records_per_family_in_directory(input_dir):
    # Získanie zoznamu všetkých CSV súborov v danom adresári, ktoré obsahujú "12ks" v názve
    all_csv_files = glob.glob(os.path.join(input_dir, "*12k*.csv"))

    # Overenie, či sa našli nejaké CSV súbory
    if not all_csv_files:
        print(f"Nenašli sa žiadne CSV súbory v adresári: {input_dir} obsahujúce '12k' v názve.")
        return
    number_of_families = 0
    # Spracovanie každého CSV súboru
    for file in all_csv_files:
        df = pd.read_csv(file)
        
        # Spočítanie počtu záznamov pre každú unikátnu hodnotu v stĺpci "Family"
        family_counts = df['Family'].value_counts()

        # Vypísanie výsledkov pre hodnoty s počtom záznamov 1000 a viac
        for family, count in family_counts.items():
            if count >= 1000:
                number_of_families+=1
                print(f"Spracovanie súboru: {file}")
                print(f"  Family: {family}, Počet záznamov: {count}")
    print(f"POCET ROZNYCH RODIN: {number_of_families}")
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prejde všetky CSV súbory obsahujúce '12k' v názve v zadanom adresári a vypíše unikátne hodnoty zo stĺpca 'Family' s počtom záznamov 1000 a viac.")
    parser.add_argument("-i", "--input_dir", required=True, help="Cesta k vstupnému adresáru s CSV súbormi.")

    args = parser.parse_args()

    count_records_per_family_in_directory(args.input_dir)
