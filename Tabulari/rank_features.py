import argparse
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif

from dataset import set_seed, load_splits


def main():
    parser = argparse.ArgumentParser(
        description="Estrazione e ordinamento delle feature per importanza (ANOVA F-Score)"
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="data/heart_disease_health_indicators.csv",
        help="Percorso del file CSV del dataset tabulare",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=15,
        help="Numero di feature top da evidenziare (default: 15)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed per la riproducibilità dello split (deve coincidere con DeepMlp.py)",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="feature_importance_anova.csv",
        help="File CSV in cui salvare la classifica completa delle feature",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    splits = load_splits(args.csv, seed=args.seed)

    #Recupera i nomi delle feature 
    df_header = pd.read_csv(args.csv, nrows=1)
    target_col = "HeartDiseaseorAttack"
    if target_col in df_header.columns:
        feature_names = df_header.drop(columns=[target_col]).columns.tolist()
    else:
        feature_names = df_header.columns[1:].tolist()

    assert len(feature_names) == splits["X_train"].shape[1], (
        f"Mismatch tra nomi colonne ({len(feature_names)}) e feature in X_train ({splits['X_train'].shape[1]})"
    )

    # Calcola i punteggi ANOVA F-value sul Training Set
    selector = SelectKBest(score_func=f_classif, k=args.k)
    selector.fit(splits["X_train"], splits["y_train"])

    #Costruisce la classifica ordinata dalla più importante alla meno importante
    ranking_df = pd.DataFrame({
        "Posizione": range(1, len(feature_names) + 1),
        "Feature": feature_names,
        "Indice_Originale": range(len(feature_names)),
        "ANOVA_F_Score": selector.scores_,
        "P_Value": selector.pvalues_,
        f"Top_{args.k}": selector.get_support(),
    })

    ranking_df = ranking_df.sort_values(by="ANOVA_F_Score", ascending=False).reset_index(drop=True)
    ranking_df["Posizione"] = ranking_df.index + 1

    print("\n" + "=" * 75)
    print(f"CLASSIFICA IMPORTANZA FEATURE (ANOVA F-SCORE) - TOP {args.k} EVIDENZIATE")
    print("=" * 75)
    print(ranking_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    selected = ranking_df[ranking_df[f"Top_{args.k}"]]["Feature"].tolist()
    discarded = ranking_df[~ranking_df[f"Top_{args.k}"]]["Feature"].tolist()

    print(f"\n[+] {len(selected)} Feature mantenute (k={args.k}):")
    print("   ", ", ".join(selected))

    print(f"\n[-] {len(discarded)} Feature scartate:")
    print("   ", ", ".join(discarded))

    ranking_df.to_csv(args.output_csv, index=False, float_format="%.6f")
    print(f"\n[+] Tabella completa salvata in: {args.output_csv}")


if __name__ == "__main__":
    main()
