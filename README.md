# computer-vision

Trabalho de Visão Computacional (UFPel): danos em pavimentos.

| Pasta | Conteúdo |
|---|---|
| `detection/` | **Detecção** (rachadura, remendo, buraco) com YOLO: pipeline completo de experimentos. Roteiro em [`detection/TRAINING_ROADMAP.md`](detection/TRAINING_ROADMAP.md). |
| `scripts/`, `results/treino*` | **Classificação** (sem dano, leve, moderado, alto) com YOLO-cls. |
| `dataset/`, `imagens/` | Dados de origem (detecção e classificação). |

## Detecção: pipeline (pavement_v3)

```bash
cd detection
pip install -r requirements.txt
python scripts/02_prepare_dataset.py --resplit                                 # dados e divisão (teste + 3 folds)
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 1.0
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 2.0
python scripts/run_experiment.py 01_baseline                                   # E1 yolo26n × yolo11n (960)
python scripts/run_experiment.py 02_models                                     # E2 tamanhos s e m
python scripts/run_experiment.py 03_imgsz                                      # E3 resolução 1280
python scripts/run_experiment.py 04_clahe                                      # E4 CLAHE 1.0 / 2.0
python scripts/run_experiment.py 05_augmentation                               # E5 aumento de dados
python scripts/run_experiment.py 05b_freeze                                    # E5b congelar camadas (overfitting)
python scripts/tune_optuna.py && python scripts/run_experiment.py 06b_optuna_best   # E6 Optuna
python scripts/run_experiment.py 07_cv && python scripts/cv_report.py          # E7 validação cruzada (3 folds)
python scripts/run_experiment.py 08_final                                      # E8 modelo final (treino, best.pt)
python scripts/final_test.py --experiment 08_final --confirm --conf cv         # teste (uma vez)
python scripts/export_report.py                                                # E9 junta os resultados em relatorio_<data>/
```

As decisões do E1 ao E6 usam o fold 0: uma mudança só é adotada se o mAP50-95 subir mais de 0,015 (`scripts/compare.py` mostra o veredito). A validação cruzada confere no fim. Detalhes, justificativas e estimativa de tempo (~3–3,5 h numa A6000) no roteiro.
