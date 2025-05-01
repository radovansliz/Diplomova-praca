import pandas as pd
import argparse

def count_family_records(input_file, family):
    # Načítanie CSV súboru
    df = pd.read_csv(input_file)
    
    # Spočítanie riadkov, ktoré majú hodnotu v stĺpci "Family" zhodujúcu sa s argumentom
    count = df[df['Family'] == family].shape[0]
    
    print(f"Počet záznamov s hodnotou '{family}' v stĺpci 'Family': {count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Spočíta počet záznamov v súbore s danou hodnotou v stĺpci 'Family'.")
    parser.add_argument("-i", "--input", required=True, help="Názov vstupného CSV súboru.")
    parser.add_argument("-f", "--family", required=True, help="Hodnota stĺpca 'Family', ktorú chcete spočítať.")

    args = parser.parse_args()

    count_family_records(args.input, args.family)
