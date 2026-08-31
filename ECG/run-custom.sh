#!/bin/bash
#SBATCH --job-name=ecg_custom
#SBATCH --account=dl-course-q2
#SBATCH --partition=dl-course-q2
#SBATCH --qos=gpu-medium
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1 --gres=shard:5632
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=PLSGCM04C22B428T@studium.unict.it
#SBATCH --output=logs_custom/job-%j.log

echo "Avvio job Custom CNN..."

mkdir -p logs_custom

# Avvio del container Apptainer globale come da documentazione
apptainer run --nv /shared/sifs/latest.sif python train_custom.py

[ $? -eq 0 ] && echo "Completato con successo!" || echo "Errore!"