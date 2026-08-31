import streamlit as st
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms, models
from ECG.train_custom import CustomECGNet

# Configurazione della pagina
st.set_page_config(page_title="Demo ML - Diagnosi Cardiologica", layout="wide")
st.title("🫀 Progetto Machine Learning: Diagnosi Cardiologica")
st.markdown("Interfaccia dimostrativa per modelli di Classificazione, Regressione e Computer Vision.")

# Creazione delle due schede principali
tab1, tab2 = st.tabs(["📊 Dati Clinici (Tabulari)", "📈 Tracciati ECG (Immagini)"])

with tab1:
    st.header("Analisi dei Parametri Clinici")
    
    col1, col2 = st.columns(2)
    with col1:
        # 1. Selezione del task e del modello
        task = st.selectbox("Seleziona il Task:", ["Classificazione (Malattia)", "Regressione (Oldpeak)"])
        modello_tab = st.selectbox("Seleziona il Modello:", ["Random Forest", "Logistic Regression", "MLP Regressor"])
        
        # 2. Caricamento Input
        st.subheader("Input Paziente")
        uploaded_csv = st.file_uploader("Carica un file CSV (singolo paziente)", type=["csv"])
        
        # 3. Ground Truth
        ground_truth_tab = st.text_input("Inserisci il Ground Truth (es. 1 per Malato, oppure 1.2 per Oldpeak):")
        
    with col2:
        st.subheader("Risultati Predizione")
        if st.button("Avvia Analisi Clinica"):
            if uploaded_csv is not None and ground_truth_tab:
                # Qui inseriremo la logica di caricamento del file e inferenza
                st.success("Modello caricato correttamente!")
                st.info(f"Modello utilizzato: {modello_tab}")
                
                # Risultato simulato (da sostituire con model(x))
                st.metric(label="Predizione del Modello", value="In elaborazione...", delta=f"Ground Truth: {ground_truth_tab}")
            else:
                st.warning("Carica un file CSV e inserisci il Ground Truth per procedere.")

with tab2:
    st.header("Analisi del Tracciato Elettrocardiografico")
    
    col3, col4 = st.columns(2)
    with col3:
        # 1. Selezione Modello CV
        modello_cv = st.selectbox("Seleziona l'Architettura:", ["Custom CNN (Baseline)", "ResNet-18 (Transfer Learning)"])
        
        # 2. Caricamento Immagine
        st.subheader("Input Tracciato")
        uploaded_img = st.file_uploader("Carica l'immagine dell'ECG", type=["png", "jpg", "jpeg"])
        
        # 3. Ground Truth
        ground_truth_cv = st.selectbox("Seleziona il Ground Truth:", ["Normale", "Infarto Miocardico", "Altra Patologia"])
        
    with col4:
        st.subheader("Analisi ed Estrazione Feature")
        if uploaded_img is not None:
            # Mostra l'immagine in input - USING use_container_width
            image = Image.open(uploaded_img)
            st.image(image, caption="ECG Caricato", use_container_width=True)
            
            if st.button("Analizza Immagine"):
                with st.spinner('Elaborazione del tracciato ECG...'):
                    # 1. Preprocessing dell'immagine (stesse trasformazioni del test_loader)
                    test_transform = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    # Converte in RGB per sicurezza ed espande la dimensione per il batch (1, C, H, W)
                    img_tensor = test_transform(image.convert('RGB')).unsqueeze(0)
                    
                    # 2. Selezione dinamica del modello in base al menu a tendina
                    if modello_cv == "ResNet-18 (Transfer Learning)":
                        modello = models.resnet18(pretrained=False)
                        num_ftrs = modello.fc.in_features
                        modello.fc = nn.Linear(num_ftrs, 2)
                        # Assicurati di copiare il nuovo .pth scaricato dal cluster in questa cartella
                        model_path = 'weight/resnet18_ecg_finetuned.pth'
                        
                    elif modello_cv == "Custom CNN (Baseline)":
                        # Assicurati di avere la classe CustomECGNet disponibile in app.py
                        modello = CustomECGNet(num_classes=2) 
                        model_path = 'weight/resnet18_ecg_finetuned.pth'
                        
                    # 3. Caricamento dei pesi e Inferenza
                    try:
                        # map_location='cpu' permette l'esecuzione fluida in locale durante la demo
                        modello.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
                        modello.eval()
                        
                        with torch.no_grad():
                            outputs = modello(img_tensor)
                            _, predicted = torch.max(outputs, 1)
                            
                        # 4. Mappatura delle classi corretta (0: Infarto, 1: Normale)
                        classi = ["Infarto Miocardico", "Normale"] 
                        classe_predetta = classi[predicted.item()]
                        
                        st.success("Estrazione feature completata!")
                        st.info(f"Rete neurale utilizzata: {modello_cv}")
                        st.metric(label="Classe Predetta", value=classe_predetta, delta=f"GT: {ground_truth_cv}")
                        
                    except FileNotFoundError:
                        st.error(f"Errore: File dei pesi non trovato nel percorso {model_path}. Assicurati di aver scaricato i file dal cluster DMI.")