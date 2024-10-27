import pandas as pd
import os
import glob

def merge_csv_files(input_dir):
    # Získanie zoznamu všetkých CSV súborov v danom adresári
    all_csv_files = glob.glob(os.path.join(input_dir, "*.csv"))

    # Overenie, či sa našli nejaké CSV súbory
    if not all_csv_files:
        print(f"Nenašli sa žiadne CSV súbory v adresári: {input_dir}")
        return

    # Spojenie všetkých CSV súborov
    df_list = []
    for file in all_csv_files:
        df = pd.read_csv(file)
        row_count = len(df)
        print(f"Nacital sa súbor '{file}' s počtom riadkov: {row_count}")
        df_list.append(df)

    # Spojenie všetkých DataFrame do jedného
    combined_df = pd.concat(df_list, ignore_index=True)

    # Vytvorenie názvu výstupného súboru na základe názvu adresára
    directory_name = os.path.basename(os.path.normpath(input_dir))
    output_file = os.path.join(input_dir, f"merged_{directory_name}.csv")

    # Uloženie do jedného veľkého CSV súboru
    combined_df.to_csv(output_file, index=False)
    print(f"Spojený súbor uložený ako: {output_file}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Spojí všetky CSV súbory v zadanom adresári do jedného veľkého CSV súboru.")
    parser.add_argument("-i", "--input_dir", required=True, help="Cesta k vstupnému adresáru s CSV súbormi.")

    args = parser.parse_args()

    merge_csv_files(args.input_dir)
