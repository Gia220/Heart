# dataset.py
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

TARGET = "HeartDiseaseorAttack"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_splits(csv_path, val_size=0.15, test_size=0.15, seed=42):
    """
    Carica il CSV, fa split train/val/test e normalizza usando SOLO le statistiche del train.
    """
    df = pd.read_csv(csv_path)


    y = df[TARGET].values.astype(np.float32)
    X = df.drop(columns=[TARGET]).values.astype(np.float32)

    # separo il test
    X_tmp, X_test, y_tmp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=seed
    )

    #divido il resto in train/val 
    rel_val = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tmp, y_tmp, test_size=rel_val, stratify=y_tmp, random_state=seed
    )

    # Scaler fittato solo sul train 
    scaler = StandardScaler().fit(X_train)
    X_train, X_val, X_test = (scaler.transform(a).astype(np.float32) for a in (X_train, X_val, X_test))

    n_pos, n_neg = y_train.sum(), len(y_train) - y_train.sum()
    print(f"Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)} | "f"% positivi train: {100 * n_pos / len(y_train):.2f}")

    return {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
        "scaler": scaler,
        "feature_names": list(df.drop(columns=[TARGET]).columns),
        "pos_weight": float(n_neg / n_pos),   # da usare in BCEWithLogitsLoss
    }


def make_loaders(splits, batch_size=1024, num_workers=0):
    """Crea i DataLoader a partire da tensori."""
    def to_ds(X, y):
        return TensorDataset(torch.from_numpy(X), torch.from_numpy(y).view(-1, 1))

    train_loader = DataLoader(to_ds(splits["X_train"], splits["y_train"]), batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(to_ds(splits["X_val"], splits["y_val"]), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(to_ds(splits["X_test"], splits["y_test"]), batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader