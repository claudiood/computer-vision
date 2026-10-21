import numpy as np
from ultralytics import YOLO

MODEL_WEIGHTS = "results/treino5_1080-yolom/weights/best.pt" 
DATASET_PATH = "dataset/class_2" 
IMGSZ = 1080 
PROJECT = "/home/andressa/Documents/UFPEL/Semestre 7/visao/computer-vision/results/treino5_1080-yolom"

SPLIT_EVAL = "test" 

print(f"--> Carregando pesos de: {MODEL_WEIGHTS}")
modelo = YOLO(MODEL_WEIGHTS)

print(f"--> Executando validação no conjunto '{SPLIT_EVAL}'...")
metricas = modelo.val(data=DATASET_PATH, imgsz=IMGSZ, split=SPLIT_EVAL, project=PROJECT) 

m = metricas.confusion_matrix.matrix  
acertos = np.diag(m)  

precisao = acertos / np.maximum(m.sum(axis=1), 1)  
recall = acertos / np.maximum(m.sum(axis=0), 1)  
f1 = 2 * precisao * recall / np.maximum(precisao + recall, 1e-9) 

print("\n" + "=" * 50)
print(f"RESULTADOS DA AVALIAÇÃO CONJUNTO: {SPLIT_EVAL.upper()}")
print(f"Acurácia Top-1 Total: {metricas.top1:.2%}")  
print(f"Acurácia Top-5 Total: {metricas.top5:.2%}")  
print("-" * 50)
print(f"{'Classe':<20} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 55)

for i, nome in modelo.names.items():  
    print(f"{nome:<20} | {precisao[i]:<10.2f} | {recall[i]:<10.2f} | {f1[i]:<10.2f}")

print("=" * 50)