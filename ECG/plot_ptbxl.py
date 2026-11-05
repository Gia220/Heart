import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import wfdb

DATASET_DIR = "../data/ptb-xl"  
SAMPLING_RATE = 500  #frequeza
SAMPLE_ECG_ID = 39


def load_ptbxl_metadata(data_dir):
    """Carica il dataframe contenente i metadati di PTB-XL."""
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"File non trovato: {csv_path}. Verifica la struttura delle cartelle."
        )

    df = pd.read_csv(csv_path, index_col="ecg_id")
    return df


def load_raw_ecg_signal(df, record_id, data_dir, sampling_rate=100):
    """Carica il segnale numerico di uno specifico record ECG (12 derivazioni)."""
    if sampling_rate == 100:
        rel_path = df.loc[record_id].filename_lr
    else:
        rel_path = df.loc[record_id].filename_hr

    full_path = os.path.join(data_dir, rel_path)

    # wfdb.rdsamp legge il file binario .dat/.hea
    signal, meta = wfdb.rdsamp(full_path)
    return signal, meta


def plot_12_lead_ecg(signal, meta, record_id, diagnostic_class="N/A"):
    """Grafica le 12 derivazioni dell'ECG in una griglia 3x4 """
    num_samples, num_channels = signal.shape
    fs = meta["fs"]
    time_axis = np.arange(num_samples) / fs
    lead_names = meta["sig_name"]

    fig, axes = plt.subplots(4, 3, figsize=(15, 10), sharex=True, sharey=True)
    fig.suptitle(
        f"ECG Record ID: {record_id} | Diagnosi: {diagnostic_class} | Freq: {fs} Hz",
        fontsize=14,
        fontweight="bold",
    )

    for i, ax in enumerate(axes.flat):
        if i < num_channels:
            ax.plot(time_axis, signal[:, i], color="navy", lw=1.2)
            ax.set_title(lead_names[i], fontsize=11, fontweight="bold")
            ax.grid(True, which="both", linestyle="--", alpha=0.5)
            ax.set_ylim([-1.5, 1.5])  # Scala mV indicativa

            if i >= 9:
                ax.set_xlabel("Tempo (s)")
            if i % 3 == 0:
                ax.set_ylabel("Ampiezza (mV)")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_overlay_ecg(signal, meta, record_id):
    """Grafica le 12 derivazioni sovrapposte in un unico tracciato per ispezione rapida."""
    num_samples, _ = signal.shape
    fs = meta["fs"]
    time_axis = np.arange(num_samples) / fs
    lead_names = meta["sig_name"]

    plt.figure(figsize=(14, 5))
    for i in range(signal.shape[1]):
        plt.plot(time_axis, signal[:, i], label=lead_names[i], alpha=0.7)

    plt.title(f"Sovrapposizione 12 Derivazioni - Record ID: {record_id}")
    plt.xlabel("Tempo (secondi)")
    plt.ylabel("Ampiezza (mV)")
    plt.legend(loc="upper right", ncol=6, fontsize="small")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()


# ==========================================
# ESECUZIONE
# ==========================================
if __name__ == "__main__":
    print("Caricamento metadati PTB-XL...")
    df_metadata = load_ptbxl_metadata(DATASET_DIR)
    print(f"Dataset caricato con successo! Totale record: {len(df_metadata)}")

    # Selezioniamo un ID di esempio (es. il primo record)
    
    sample_scp_codes = df_metadata.loc[SAMPLE_ECG_ID, "scp_codes"]

    print(f"\nCaricamento segnale per ECG ID: {SAMPLE_ECG_ID}...")
    ecg_signal, metadata = load_raw_ecg_signal(df_metadata, SAMPLE_ECG_ID, DATASET_DIR, sampling_rate=SAMPLING_RATE)

    print(f"Dimensione matrice segnale: {ecg_signal.shape} (Campioni x Derivazioni)")
    print(f"Frequenza di campionamento: {metadata['fs']} Hz")
    print(f"Nomi derivazioni: {metadata['sig_name']}")

    # Plot delle 12 derivazioni in griglia
    plot_12_lead_ecg(
        ecg_signal, metadata, record_id=SAMPLE_ECG_ID, diagnostic_class=sample_scp_codes
    )

    # Plot sovrapposto
    plot_overlay_ecg(ecg_signal, metadata, record_id=SAMPLE_ECG_ID)