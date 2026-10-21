import numpy as np
from ultralytics import YOLO

DATASET_PATH = "dataset/class_2" 

MODEL_NAME = "yolo26m-cls.pt"  
EPOCHS = 30  
IMGSZ = 1080  
BATCH = 8  
PROJECT_DIR = "/home/andressa/Documents/UFPEL/Semestre 7/visao/computer-vision/results"
EXP_NAME = "treino5_1080-yolom"

print("--> Iniciando o treinamento...")
modelo = YOLO(MODEL_NAME)

modelo.train(
    data=DATASET_PATH,
    epochs=EPOCHS,
    imgsz=IMGSZ,
    batch=BATCH,
    patience=10, 
    project=PROJECT_DIR,  
    name=EXP_NAME,  
    exist_ok=True, 
)

print("\n" + "=" * 50)
print("--> Avaliando o melhor modelo carregado (best.pt)...")
caminho_best_weights = f"{PROJECT_DIR}/{EXP_NAME}/weights/best.pt" 
melhor_modelo = YOLO(caminho_best_weights)

metricas = melhor_modelo.val(data=DATASET_PATH, imgsz=IMGSZ, project=PROJECT_DIR)

print("\n" + "=" * 50)
print("RESUMO DAS MÉTRICAS GERAIS:")
print(f"Top-1 Accuracy : {metricas.top1:.2%}") 
print(f"Top-5 Accuracy : {metricas.top5:.2%}")  
print("=" * 50)

m = metricas.confusion_matrix.matrix  
acertos = np.diag(m)  

precisao = acertos / np.maximum(m.sum(axis=1), 1)  
recall = acertos / np.maximum(m.sum(axis=0), 1)  
f1 = 2 * precisao * recall / np.maximum(precisao + recall, 1e-9)  

print(f"{'Classe':<20} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10} | {'Imagens':<8}")
print("-" * 68)

for i, nome in melhor_modelo.names.items():  
    qtd_imagens = int(m[:, i].sum())  
    print(
        f"{nome:<20} | {precisao[i]:<10.2f} | {recall[i]:<10.2f} | {f1[i]:<10.2f} | {qtd_imagens:<8}"
    ) 

print("=" * 50)