import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from sklearn.preprocessing import StandardScaler
import joblib

# ==========================================
# IMPORT DEI MODELLI CUSTOM
# ==========================================
from ECG.train_custom import CustomECGNet
from Classificatore.RegressioneLineare import LogisticRegressor
# Sostituisci "nome_file_mlp" con il nome reale del tuo script Python (es. Classificatore.mlp)
from Classificatore.MLP import HeartDiseaseMLP 

# ==========================================
# FUNZIONI DI CACHE PER I TRASFORMATORI
# ==========================================
@st.cache_resource
def get_fitted_scaler(csv_path="data/heart_johnsmith88_mod.csv"):
    df = pd.read_csv(csv_path)
    df = df.apply(pd.to_numeric, errors='coerce').dropna()
    X_raw = df.iloc[:, :-1].values
    scaler = StandardScaler()
    scaler.fit(X_raw)
    return scaler

@st.cache_resource
def load_poly_transformers():
    poly = joblib.load('weight/poly_transform.pkl')
    poly_scaler = joblib.load('weight/poly_scaler.pkl')
    return poly, poly_scaler


# ==========================================
# CONFIGURAZIONE PAGINA
# ==========================================
st.set_page_config(page_title="Demo ML - Diagnosi Cardiologica", layout="wide")
st.title("🫀 Progetto Machine Learning: Diagnosi Cardiologica")
st.markdown("Interfaccia dimostrativa per modelli di Classificazione, Regressione e Computer Vision.")

# Creazione delle due schede principali
tab1, tab2 = st.tabs(["📊 Dati Clinici (Tabulari)", "📈 Tracciati ECG (Immagini)"])

# ==========================================
# TAB 1: DATI TABULARI (CONFRONTO MODELLI)
# ==========================================

with tab1:
    st.header("Analisi dei Parametri Clinici - Confronto Modelli")
    
    default_input = "54,1,0,120,188,0,1,113,0,1.4,1,1,3,0"
    user_input_str = st.text_input(
        "Inserisci i parametri (valori separati da virgola):", 
        value=default_input,
        help="Inserisci 13 feature cliniche + l'eventuale Ground Truth come 14° valore."
    )
    
    st.markdown("---")
    
    if st.button("Avvia Inferenza Comparativa (Dati Clinici)"):
        try:
            valori = [float(val.strip()) for val in user_input_str.split(",") if val.strip() != ""]
            
            if len(valori) < 13:
                st.error(f"Formato non valido: rilevati {len(valori)} valori. Sono richieste almeno 13 feature.")
            else:
                features_raw = np.array(valori[:13]).reshape(1, -1)
                ground_truth = int(valori[13]) if len(valori) >= 14 else None
                classi_cliniche = {0: "Basso Rischio", 1: "Alto Rischio"}

                # --- SCALER CONDIVISO ---
                scaler_lin = get_fitted_scaler()
                features_lin = scaler_lin.transform(features_raw)
                tensor_lin = torch.tensor(features_lin, dtype=torch.float32)

                # 1. Modello Lineare (PyTorch)
                model_lin = LogisticRegressor(in_features=13)
                model_lin.load_state_dict(torch.load('weight/linear_regressor.pth', map_location='cpu'))
                model_lin.eval()
                with torch.no_grad():
                    prob_lin = torch.sigmoid(model_lin(tensor_lin)).item()
                    pred_lin = 1 if prob_lin >= 0.5 else 0
                    
                # 2. Modello MLP (PyTorch)
                model_mlp = HeartDiseaseMLP(in_features=13)
                model_mlp.load_state_dict(torch.load('weight/MLP.pth', map_location='cpu'))
                model_mlp.eval()
                with torch.no_grad():
                    prob_mlp = torch.sigmoid(model_mlp(tensor_lin)).item()
                    pred_mlp = 1 if prob_mlp >= 0.5 else 0
                    
                # 3. Random Forest (Scikit-Learn)
                rf_model = joblib.load('weight/random_forest.pkl')
                # predict_proba restituisce [prob_classe_0, prob_classe_1]
                prob_rf = rf_model.predict_proba(features_lin)[0][1] 
                pred_rf = rf_model.predict(features_lin)[0]

                # --- ELABORAZIONE POLINOMIALE ---
                poly_transformer, poly_scaler = load_poly_transformers()
                features_poly_raw = poly_transformer.transform(features_lin) 
                poly_dim = features_poly_raw.shape[1]
                features_poly = poly_scaler.transform(features_poly_raw)
                tensor_poly = torch.tensor(features_poly, dtype=torch.float32)

                # 4. Modello Poly Standard
                model_poly = LogisticRegressor(in_features=poly_dim)
                model_poly.load_state_dict(torch.load('weight/poly_regressor.pth', map_location='cpu'))
                model_poly.eval()
                with torch.no_grad():
                    prob_poly = torch.sigmoid(model_poly(tensor_poly)).item()
                    pred_poly = 1 if prob_poly >= 0.5 else 0


                # --- VISUALIZZAZIONE RISULTATI ---
                if ground_truth is not None:
                    st.info(f"**Verità Clinica (Ground Truth):** {classi_cliniche[ground_truth]}")
                
                # Layout a 5 colonne
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.subheader("Lineare")
                    st.metric(label="Diagnosi", value=classi_cliniche[pred_lin])
                    st.metric(label="Probabilità", value=f"{prob_lin * 100:.1f}%")
                    if ground_truth is not None:
                        if pred_lin == ground_truth:
                            st.success("✅")
                        else:
                            st.error("❌")
                            
                with col2:
                    st.subheader("Poly (Std)")
                    st.metric(label="Diagnosi", value=classi_cliniche[pred_poly])
                    st.metric(label="Probabilità", value=f"{prob_poly * 100:.1f}%")
                    if ground_truth is not None:
                        if pred_poly == ground_truth:
                            st.success("✅")
                        else:
                            st.error("❌")
                            
                with col3:
                    st.subheader("Rete MLP")
                    st.metric(label="Diagnosi", value=classi_cliniche[pred_mlp])
                    st.metric(label="Probabilità", value=f"{prob_mlp * 100:.1f}%")
                    if ground_truth is not None:
                        if pred_mlp == ground_truth:
                            st.success("✅")
                        else:
                            st.error("❌")

                with col4:
                    st.subheader("Random Forest")
                    st.metric(label="Diagnosi", value=classi_cliniche[pred_rf])
                    st.metric(label="Probabilità", value=f"{prob_rf * 100:.1f}%")
                    if ground_truth is not None:
                        if pred_rf == ground_truth:
                            st.success("✅")
                        else:
                            st.error("❌")

        except ValueError as e:
            st.error(f"Errore di parsing: {e}")
        except FileNotFoundError as e:
            st.error(f"File non trovato! Assicurati di aver generato tutti i pesi (.pth e .pkl). Dettagli: {e}")
# ==========================================
# TAB 2: IMMAGINI ECG (CONFRONTO MODELLI)
# ==========================================
with tab2:
    st.header("Analisi del Tracciato Elettrocardiografico - Confronto Architetture")
    
    col_input, col_gt = st.columns(2)
    with col_input:
        uploaded_img = st.file_uploader("Carica l'immagine dell'ECG", type=["png", "jpg", "jpeg"])
    with col_gt:
        ground_truth_cv = st.selectbox("Seleziona il Ground Truth:", ["Normale", "Infarto Miocardico", "Altra Patologia"])
        
    if uploaded_img is not None:
        image = Image.open(uploaded_img)
        st.image(image, caption="ECG Caricato", width=400)
        
        st.markdown("---")
        
        if st.button("Avvia Inferenza Comparativa (ECG)"):
            with st.spinner('Elaborazione in corso sulle reti neurali...'):
                test_transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
                img_tensor = test_transform(image.convert('RGB')).unsqueeze(0)
                classi = ["Infarto Miocardico", "Normale"] 
                
                try:
                    # --- RETE 1: Custom CNN ---
                    modello_custom = CustomECGNet(num_classes=2) 
                    modello_custom.load_state_dict(torch.load('weight/custom_ecg_net_final.pth', map_location='cpu'))
                    modello_custom.eval()
                    with torch.no_grad():
                        out_custom = modello_custom(img_tensor)
                        _, pred_custom = torch.max(out_custom, 1)
                        classe_custom = classi[pred_custom.item()]
                        
                    # --- RETE 2: ResNet-18 ---
                    modello_resnet = models.resnet18(pretrained=False)
                    modello_resnet.fc = nn.Linear(modello_resnet.fc.in_features, 2)
                    modello_resnet.load_state_dict(torch.load('weight/resnet18_ecg_finetuned.pth', map_location='cpu'))
                    modello_resnet.eval()
                    with torch.no_grad():
                        out_resnet = modello_resnet(img_tensor)
                        _, pred_resnet = torch.max(out_resnet, 1)
                        classe_resnet = classi[pred_resnet.item()]
                        
                    col_net1, col_net2 = st.columns(2)
                    
                    with col_net1:
                        st.subheader("Custom CNN (Baseline)")
                        st.metric(label="Diagnosi Predetta", value=classe_custom)
                        if ground_truth_cv in classi:
                            if classe_custom == ground_truth_cv:
                                st.success("✅ **CORRETTO!**")
                            else:
                                st.error(f"❌ **ERRATO!** Era {ground_truth_cv}.")
                                
                    with col_net2:
                        st.subheader("ResNet-18 (Transfer Learning)")
                        st.metric(label="Diagnosi Predetta", value=classe_resnet)
                        if ground_truth_cv in classi:
                            if classe_resnet == ground_truth_cv:
                                st.success("✅ **CORRETTO!**")
                            else:
                                st.error(f"❌ **ERRATO!** Era {ground_truth_cv}.")
                                
                    if ground_truth_cv == "Altra Patologia":
                        st.warning("⚠️ Hai inserito 'Altra Patologia'. I modelli sono addestrati solo per Normale/Infarto.")

                except FileNotFoundError as e:
                    st.error(f"Errore: Impossibile trovare i file dei pesi. Assicurati che i .pth siano nella cartella 'weight'. ({e})")