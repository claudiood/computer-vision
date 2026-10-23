import gc

import optuna
import torch
from ultralytics import YOLO

DATASET_PATH = './dataset/class' #selecionem o dataset que querem treinar aqui

MODEL_NAME = "yolo26m-cls.pt" #mudem o modelo aqui

EXP_NAME = "optuna_yolom" #aqui é o nome do experimento, mudem pra saber qual treino é qual

def objective(trial):
    lr0 = trial.suggest_float("lr0", 1e-5, 1e-2, log=True)
    imgsz = trial.suggest_categorical("imgsz", [640, 896, 1080])

    model = YOLO(MODEL_NAME)
    metrics = model.train(
        data=DATASET_PATH,
        epochs=100,
        imgsz=imgsz,
        batch=4,
        optimizer="Adam",
        lr0=lr0,
        patience=10,
        project=f"./results/{EXP_NAME}",
        name="search",
        exist_ok=True,
    )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return metrics.top1


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

best_params = study.best_trial.params
print(f"\nBest hyperparameters: {best_params}")

best_model = YOLO(MODEL_NAME)

best_model.train(
    data=DATASET_PATH,
    epochs=100,
    imgsz=best_params["imgsz"],
    batch=4,
    optimizer="Adam",
    lr0=best_params["lr0"],
    patience=10,
    project=f"./results/{EXP_NAME}",
    name="final",
    exist_ok=True,
)
metrics = best_model.val(data=DATASET_PATH, imgsz=best_params["imgsz"], split="test", project='./results')

print(f"\nTest accuracy: {metrics.top1:.2%}")
