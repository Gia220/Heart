import pandas as pd

try:
    results_df = pd.read_csv("results.csv")
    
    print("--- CLASSIFICA MODELLI ---")
    # Usa print() al posto di display()
    print(results_df.sort_values(by="F1_Score", ascending=False).to_string(index=False))
    
except FileNotFoundError:
    print("Il file results.csv non esiste ancora. Addestra almeno un modello!")