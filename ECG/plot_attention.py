import os
import ast
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import wfdb
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import torch.nn.functional as F

from transformers import ECGTransformer
from dataset_ptbxl import remove_baseline_wander

def extract_attention(model, x):
    """
    Estrae la matrice di attenzione dall'ultimo layer del Transformer.
    x shape: [1, 12, 1000]
    """
    model.eval()
    with torch.no_grad():
        feat = model.conv_stem(x)
        feat = feat.permute(0, 2, 1)
        feat = feat + model.pos_encoder
        
        # Passaggio nei primi layer
        for i in range(len(model.transformer_encoder.layers) - 1):
            feat = model.transformer_encoder.layers[i](feat)
            
        # Estrazione pesi dall'ultimo blocco di Self-Attention
        last_layer = model.transformer_encoder.layers[-1]
        _, attn_weights = last_layer.self_attn(feat, feat, feat, need_weights=True)
        
        return attn_weights.squeeze(0) # [250, 250]


def load_ptbxl_sample(data_dir, target_class="MI", ecg_id=None):
    """
    Carica i metadati e un segnale reale a 100 Hz (1000 timesteps) da PTB-XL.
    Risolve il problema dei percorsi nidificati errati.
    """
    csv_path = os.path.join(data_dir, "ptbxl_database.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Impossibile trovare ptbxl_database.csv in: {data_dir}")
        
    df = pd.read_csv(csv_path, index_col='ecg_id')
    df.scp_codes = df.scp_codes.apply(lambda x: ast.literal_eval(x))

    # Carica la mappatura delle 5 super-classi
    statements_path = os.path.join(data_dir, "scp_statements.csv")
    agg_df = pd.read_csv(statements_path, index_col=0)
    agg_df = agg_df[agg_df.diagnostic == 1]

    def aggregate_diagnostic(y_dic):
        tmp = []
        for key in y_dic.keys():
            if key in agg_df.index:
                tmp.append(agg_df.loc[key].diagnostic_class)
        return list(set(tmp))

    df['diagnostic_superclass'] = df.scp_codes.apply(aggregate_diagnostic)

    # Selezione del record
    if ecg_id is not None and ecg_id in df.index:
        selected_row = df.loc[ecg_id]
    else:
        # Trova il primo record con la classe desiderata (es. MI)
        selected_df = df[df.diagnostic_superclass.apply(lambda x: target_class in x)]
        if len(selected_df) == 0:
            selected_row = df.iloc[0]
        else:
            selected_row = selected_df.iloc[0]

    relative_path = selected_row['filename_lr'] 
    full_path = os.path.join(data_dir, relative_path)
    
    # Caricamento con wfdb
    signal, meta = wfdb.rdsamp(full_path)

    #per baseline_wander
    signal = signal.T.astype(np.float32)
    signal = remove_baseline_wander(signal, fs=100, cutoff=0.5)
    
    # Trasposizione a [12, 1000] per PyTorch
    signal_tensor = torch.tensor(signal, dtype=torch.float32)
    
    return signal_tensor, selected_row.name, selected_row['diagnostic_superclass']


def plot_ecg_with_attention(signal, attn_matrix, ecg_id, labels, lead_idx=1, lead_name="DII"):
    """
    Grafica il segnale ECG di una derivazione specifica con sovrapposta la heatmap dell'attenzione.
    """
    # Media dell'attenzione ricevuta da ciascun token temporale (da [250, 250] -> [250])
    attn_1d = attn_matrix.mean(dim=0).numpy()
    
    # Interpolazione lineare da 250 a 1000 timesteps per allinearla al tracciato originale
    attn_1d_resized = F.interpolate(
        torch.tensor(attn_1d).unsqueeze(0).unsqueeze(0),
        size=1000,
        mode='linear',
        align_corners=False
    ).squeeze().numpy()
    
    # Normalizzazione [0, 1] per la scala cromatica
    attn_1d_resized = (attn_1d_resized - attn_1d_resized.min()) / (attn_1d_resized.max() - attn_1d_resized.min() + 1e-8)
    
    # Estrazione della derivazione selezionata (es. DII o V2/V3)
    ecg_lead = signal[lead_idx, :].numpy()
    time_axis = np.linspace(0, 10, 1000) # 10 secondi
    
    fig, ax = plt.subplots(figsize=(14, 5))
    
    # 1. Tracciato ECG
    ax.plot(time_axis, ecg_lead, color='black', linewidth=1.3, label=f'Segnale ECG - Derivazione {lead_name}')
    
    # 2. Linea Isoelettrica di riferimento (utile per identificare il sopraslivellamento ST)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, label='Linea Isoelettrica')
    
    # 3. Mappa di Attenzione (Heatmap di sfondo)
    extent = [0, 10, ecg_lead.min() - 0.2, ecg_lead.max() + 0.2]
    im = ax.imshow(
        attn_1d_resized[np.newaxis, :], 
        cmap='Reds', 
        aspect='auto', 
        extent=extent, 
        alpha=0.55
    )
    
    # Formattazione Grafica
    cbar = plt.colorbar(im, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_label('Intensità dell\'Attenzione (Focus del Modello)', rotation=270, labelpad=15)
    
    ax.set_title(f"Visualizzazione Attenzione Transformer - ECG ID: {ecg_id} | Diagnosi: {labels}", fontsize=13, fontweight='bold')
    ax.set_xlabel("Tempo (secondi)", fontsize=11)
    ax.set_ylabel("Ampiezza (mV)", fontsize=11)
    ax.set_ylim(extent[2], extent[3])
    ax.legend(loc='upper right')
    ax.grid(True, which='both', linestyle=':', alpha=0.4)
    
    plt.tight_layout()
    plt.show()


def plot_ecg_clinical_standard(signal, attn_matrix, ecg_id, labels, lead_idx=1, lead_name="DII"):
    """
    Disegna l'ECG rispettando la griglia clinica standard (carta millimetrata rosa)
    e sovrappone la mappa di attenzione del Transformer.
    """
    # 1. Preparazione dell'Attenzione
    attn_1d = attn_matrix.mean(dim=0).numpy()
    attn_1d_resized = F.interpolate(
        torch.tensor(attn_1d).unsqueeze(0).unsqueeze(0),
        size=1000,
        mode='linear',
        align_corners=False
    ).squeeze().numpy()
    attn_1d_resized = (attn_1d_resized - attn_1d_resized.min()) / (attn_1d_resized.max() - attn_1d_resized.min() + 1e-8)

    # 2. Dati del segnale (10 secondi a 100 Hz)
    ecg_lead = signal[lead_idx, :].numpy()
    time_axis = np.linspace(0, 10, 1000)

    # Allargato a 16 pollici di larghezza (senza colorbar occuperà tutto lo spazio)
    fig, ax = plt.subplots(figsize=(16, 5), dpi=150)
    
    # 3. IMPOSTAZIONE CARTA MILLIMETRATA CLINICA
    ax.set_facecolor('#FFF5F5')
    
    # Ampiezza fissa da -1 a 1 mV
    y_min, y_max = -1.0, 1.0
    
    # Griglia Piccola (40 ms = 0.04s, 0.1 mV)
    ax.set_xticks(np.arange(0, 10.04, 0.04), minor=True)
    ax.set_yticks(np.arange(y_min, y_max + 0.1, 0.1), minor=True)
    ax.grid(True, which='minor', color='#FFC0CB', linestyle='-', linewidth=0.5, alpha=0.7)
    
    # Griglia Grande (200 ms = 0.2s, 0.5 mV)
    ax.set_xticks(np.arange(0, 10.2, 0.2))
    ax.set_yticks(np.arange(y_min, y_max + 0.5, 0.5))
    ax.grid(True, which='major', color='#FF8080', linestyle='-', linewidth=1.0, alpha=0.9)

    # FIX ASSE X: Mostra l'etichetta di testo solo per i secondi interi (0, 1, 2... 10)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, pos: f"{int(x)}" if x % 1 == 0 else ""))

    # 4. SOVRAPPOSIZIONE MAPPA DI ATTENZIONE
    extent = [0, 10, y_min, y_max]
    
    im = ax.imshow(
        attn_1d_resized[np.newaxis, :], 
        cmap='YlOrRd', 
        aspect='auto', 
        extent=extent, 
        alpha=0.45,
        zorder=2
    )

    # 5. TRACCIATO ECG
    ax.plot(time_axis, ecg_lead, color='#000080', linewidth=1.4, label=f'Derivazione {lead_name}', zorder=3)
    ax.axhline(0, color='black', linestyle='-', linewidth=0.8, alpha=0.6, zorder=3)

    # Formattazione finale (Colorbar rimossa)
    ax.set_title(f"Standard Clinical ECG & Attention Overlay | Patient ID: #{ecg_id} | Diagnosi: {labels}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Tempo (secondi) [25 mm/s]", fontsize=12)
    ax.set_ylabel("Ampiezza (mV) [10 mm/mV]", fontsize=12)
    ax.set_xlim(0, 10)
    ax.set_ylim(y_min, y_max)
    ax.legend(loc='upper right', framealpha=0.9)

    plt.tight_layout()
    plt.show()
# ==========================================
# 5. MAIN EXECUTION
# ==========================================
if __name__ == "__main__":

    data_dir = "../data/ptb-xl"
    model_path = "../weight/transformer_ecg_best_2.pth"
    ecg_id = 10
    lead_idx = 1
    
    # Nomi delle 12 derivazioni nello standard PTB-XL
    LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    
    # 1. Carica il modello e i pesi
    device = torch.device("cpu")
    model = ECGTransformer()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"[+] Pesi del modello caricati con successo da: {model_path}")
    else:
        print(f"[!] Warning: Pesi non trovati in {model_path}. Verrà usata l'inizializzazione casuale per testare la grafica.")
        
    # 2. Carica il segnale ECG di un paziente reale
    signal_tensor, ecg_id, labels = load_ptbxl_sample(data_dir, target_class="MI")#, ecg_id=ecg_id)
    print(f"[+] Caricato record ECG ID #{ecg_id} con diagnosi: {labels}")
    
    # 3. Estrai la matrice di attenzione
    x_input = signal_tensor.unsqueeze(0) # [1, 12, 1000]
    attn_matrix = extract_attention(model, x_input)
    
    # 4. Genera il plot per la derivazione scelta
    lead_name = LEAD_NAMES[lead_idx]
    #plot_ecg_with_attention(signal_tensor, attn_matrix, ecg_id, labels, lead_idx=lead_idx, lead_name=lead_name)
    plot_ecg_clinical_standard(signal_tensor, attn_matrix, ecg_id, labels, lead_idx=lead_idx, lead_name=lead_name)