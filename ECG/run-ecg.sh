#!/bin/bash
#SBATCH --job-name=ecg_resnet_training
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1 --gres=shard:8000
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=PLSGCM04C22B428T@studium.unict.it
#SBATCH --output=logs/job-%j.log

echo "Inizio allocazione per il progetto di classificazione ECG..."

# Creiamo la cartella per i log di testo se non esiste
mkdir -p logs

# Avviamo il container Apptainer con il nostro script Python
# Utilizziamo l'immagine preconfigurata indicata nelle linee guida del DMI
apptainer run --nv /shared/sifs/latest.sif python train_ecg.py --batch_size 128 --epochs 15 --output_dir ./risultati_ecg

[ $? -eq 0 ] && echo "Esecuzione completata con successo!" || echo "Errore durante l'esecuzione del job."