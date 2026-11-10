import ast
import os
import numpy as np
import pandas as pd
import torch
import wfdb
from torch.utils.data import DataLoader, Dataset
from scipy.signal import butter, sosfiltfilt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             multilabel_confusion_matrix, roc_auc_score, average_precision_score)


# Mappatura delle 5 super-classi diagnostiche
SUPERCLASSES = ['NORM', 'MI', 'STTC', 'CD', 'HYP']
CLASS_TO_IDX = {cls_name: idx for idx, cls_name in enumerate(SUPERCLASSES)}


def compute_metrics(y_true, probs, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    return {
        "Threshold": threshold,
        "Accuracy": accuracy_score(y_true, preds),
        "Precision": precision_score(y_true, preds, average='macro', zero_division=0),
        "Recall": recall_score(y_true, preds, average='macro', zero_division=0),
        "F1_Score": f1_score(y_true, preds, average='macro', zero_division=0),
        "ROC_AUC": roc_auc_score(y_true, probs, average='macro'),
        "PR_AUC": average_precision_score(y_true, probs, average='macro'),
    }


def remove_baseline_wander(ecg_signal, fs=100, cutoff=2.8):
    """
    Rimuove la fluttuazione della linea isoelettrica (respiro).
    Si aspetta un array numpy di forma [12, timesteps] (canali, timesteps).
    Usa 'sosfiltfilt' per garantire massima stabilità numerica e fase zero (nessun errore temporale sui picchi).
    """
    sos = butter(N=2, Wn=cutoff, btype='highpass', fs=fs, output='sos')
    filtered_signal = sosfiltfilt(sos, ecg_signal, axis=1)
    return filtered_signal.astype(np.float32)


class PTBXLDataset(Dataset):
    """Dataset PyTorch per la classificazione a 5 super-classi su PTB-XL."""

    def __init__(self, df, data_dir, sampling_rate=100, apply_filter=False):
        self.df = df.reset_index(drop=True)
        self.data_dir = data_dir
        self.sampling_rate = sampling_rate
        self.apply_filter = apply_filter

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # Selezione del percorso in base alla frequenza di campionamento (100 Hz vs 500 Hz)
        rel_path = (
            row.filename_lr if self.sampling_rate == 100 else row.filename_hr
        )
        full_path = os.path.join(self.data_dir, rel_path)

        # Caricamento del segnale numerico reale via wfdb
        signal, _ = wfdb.rdsamp(full_path)

        # Transposizione da [1000, 12] o [5000, 12] a [12, 1000] (o [12, 5000]) per le convoluzioni 1D PyTorch:
        # Forma: [Canali/Derivazioni, Timesteps]
        signal = signal.T.astype(np.float32)

        # ---> RIMOZIONE DELLA BASELINE WANDER (OPZIONALE PER ABLATION STUDY) <---
        if self.apply_filter:
            signal = remove_baseline_wander(signal, fs=self.sampling_rate, cutoff=0.5)

        # Vettore di etichette Multi-Hot per le 5 classi (es. [1, 0, 0, 1, 0])
        labels = row.label_vector.astype(np.float32)

        return torch.tensor(signal, dtype=torch.float32), torch.tensor(
            labels, dtype=torch.float32
        )


def prepare_ptbxl_5class_data(data_dir):
    """Carica metadati, mappa le 5 super-classi SCP e prepara i vettori di etichette."""
    # 1. Caricamento database e mapping SCP
    df = pd.read_csv(
        os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id"
    )
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    scp_df = pd.read_csv(
        os.path.join(data_dir, "scp_statements.csv"), index_col=0
    )
    scp_df = scp_df[scp_df.diagnostic == 1]  # Filtra solo i codici diagnostici

    def get_superclasses(scp_dict):
        """Estrae l'insieme delle 5 super-classi diagnostiche presenti nel record."""
        super_classes = set()
        for code in scp_dict.keys():
            if code in scp_df.index:
                diagnostic_class = scp_df.loc[code].diagnostic_class
                if diagnostic_class in SUPERCLASSES:
                    super_classes.add(diagnostic_class)
        return super_classes

    df["diagnostic_superclass"] = df.scp_codes.apply(get_superclasses)

    # 2. Creazione del vettore One-Hot / Multi-Hot a 5 elementi
    def encode_labels(class_set):
        vector = np.zeros(len(SUPERCLASSES), dtype=np.float32)
        for cls in class_set:
            vector[CLASS_TO_IDX[cls]] = 1.0
        return vector

    # Filtriamo i record che hanno almeno una delle 5 super-classi
    df["has_superclass"] = df.diagnostic_superclass.apply(lambda x: len(x) > 0)
    df_filtered = df[df.has_superclass].copy()
    df_filtered["label_vector"] = df_filtered.diagnostic_superclass.apply(
        encode_labels
    )

    print(f"Dataset PTB-XL caricato con successo!")
    print(
        f" - Record con almeno una delle 5 super-classi: {len(df_filtered)} / {len(df)}"
    )
    print("\nDistribuzione delle 5 super-classi:")
    for idx, cls_name in enumerate(SUPERCLASSES):
        count = sum(df_filtered.label_vector.apply(lambda v: v[idx] == 1.0))
        print(f" - {cls_name:<5}: {count} istanze")

    # 3. Split Train / Validation / Test basato sulle pieghe ufficiali 'strat_fold'
    train_df = df_filtered[df_filtered.strat_fold <= 8]
    val_df = df_filtered[df_filtered.strat_fold == 9]
    test_df = df_filtered[df_filtered.strat_fold == 10]

    return train_df, val_df, test_df


# ==========================================
# ESECUZIONE E VERIFICA DATALOADER
# ==========================================
if __name__ == "__main__":
    DATASET_DIR = "data/ptb-xl"
    BATCH_SIZE = 32

    train_df, val_df, test_df = prepare_ptbxl_5class_data(DATASET_DIR)

    # Creazione Dataset (passa apply_filter=True per testare la pulizia del segnale)
    train_dataset = PTBXLDataset(train_df, DATASET_DIR, apply_filter=False)
    test_dataset = PTBXLDataset(test_df, DATASET_DIR, apply_filter=False)

    # Creazione DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE, shuffle=False
    )

    # Test di caricamento di un batch
    inputs, labels = next(iter(train_loader))
    print(f"\nVerifica Batch PyTorch:")
    print(
        f" - Dimensioni Input Tensor : {inputs.shape}  -> [Batch, Canali (12), Timesteps (1000)]"
    )
    print(
        f" - Dimensioni Labels Tensor: {labels.shape}  -> [Batch, Classi (5)]"
    )
    print(f" - Esempio etichetta primo campione: {labels[0].numpy()}")