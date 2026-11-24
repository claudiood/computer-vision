# computer-vision

Trabalho de Visão Computacional (UFPel): danos em pavimentos.

| Pasta | Conteúdo |
|---|---|
| `detection/` | **Detecção** (rachadura, remendo, buraco) com YOLO: pipeline completo de experimentos. Roteiro em [`detection/TRAINING_ROADMAP.md`](detection/TRAINING_ROADMAP.md). |
| `scripts/`, `results/treino*` | **Classificação** (sem dano, leve, moderado, alto) com YOLO-cls. |
| `dataset/`, `imagens/` | Dados de origem (detecção e classificação). |

## Detecção: pipeline (pavement_v3, nova rotulação)

```bash
cd detection
pip install -r requirements.txt
python scripts/02_prepare_dataset.py --resplit                                 # dados e divisão
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 1.0
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 2.0
python scripts/run_experiment.py 01_baseline                                   # E1 yolo26n 960
python scripts/run_experiment.py 02_models                                     # E2 yolo26s, yolo11n
python scripts/run_experiment.py 03_imgsz                                      # E3 1280
python scripts/run_experiment.py 04_clahe                                      # E4 CLAHE 1.0 / 2.0
python scripts/tune_optuna.py                                                  # E5 Optuna
python scripts/run_experiment.py 06_ablation                                   # E6 ablação
python scripts/run_experiment.py 07_cv && python scripts/cv_report.py          # E7 validação cruzada
python scripts/run_experiment.py 08_final                                      # E8 treino definitivo
python scripts/final_test.py --experiment 08_final --confirm --conf cv         # teste (uma vez)
```

Todas as decisões usam os mesmos 3 folds e comparação pareada (`scripts/compare.py`). Detalhes, justificativas e estimativa de tempo (~5 h numa A6000) no roteiro.
