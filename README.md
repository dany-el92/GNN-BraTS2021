# GNN-BRaTs2021
Questa repository contiene il progetto di tesi svolto da Daniela Amendola e [@Andrea Basile](https://github.com/AndreaBasile97), presso l'Università degli Studi di Bari "Aldo Moro".

## Requirements
torch

nibabel==3.2

dgl

matplotlib

medpy

python-dotenv

## Setup
Per avviare lo script per preprocessare i dati scrivere nel terminale il comando:

python -m scripts.preprocess_dataset -d ~/project_data/BraTS21_data/raw/train -n 15000 -k 0 -b 0.1 -o ~/project_data/BraTS21_data/processed/train -l _seg.nii.gz -p BraTS2021

## Dataset
Il dataset utilizzato è BraTS2021, è possibile scaricarlo al seguente link: https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1

## Dataset Pickle 
È possibile scaricare i file pickle contenenti i file dei pazienti [grafo (in formato DGL), feature, label, id] al seguente link: https://drive.google.com/file/d/152HqhaAEdtEBbyv1tYkDorVhudieyv6Y/view?usp=sharing

## File .env
È necessario creare un file .env con al suo interno:

DATASET_PICKLE_PATH = '' #file pickle con il dataset preprocessato

DATASET_PATH = '' #la cartella contenente tutti i pazienti con tutti i loro dati (grafo, label, ecc...)

MODEL_PATH = '' #modello generato da testare nella fase di test

METRICS_TRAINING_SAVE_PATH = '' #dove salvare le metriche dell'addestramento

METRICS_TESTING_SAVE_PATH = '' #dove salvare le metriche del test
