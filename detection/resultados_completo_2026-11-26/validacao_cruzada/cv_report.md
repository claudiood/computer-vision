# Validação cruzada — 3 folds

Configuração: `01_baseline/yolo26n` (modelo yolo26n.pt, imgsz 960, dataset pavement_v3, oversample None).

| fold | mAP50_95 | mAP50 | P | R | F1 | F1_op | best_conf | melhor época |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.182 | 0.353 | 0.430 | 0.335 | 0.372 | 0.376 | 0.221 | 43 |
| 1 | 0.128 | 0.275 | 0.526 | 0.256 | 0.344 | 0.343 | 0.384 | 83 |
| 2 | 0.203 | 0.416 | 0.451 | 0.444 | 0.445 | 0.438 | 0.148 | 74 |
| **média ± desvio** | **0.171 ± 0.039** | **0.348 ± 0.071** | **0.469 ± 0.050** | **0.345 ± 0.094** | **0.387 ± 0.052** | **0.386 ± 0.048** | **0.251 ± 0.121** | |

## Por classe (média ± desvio entre folds)

| métrica | média | desvio |
|---|---|---|
| F1__buraco | 0.524 | 0.045 |
| F1__rachadura | 0.276 | 0.092 |
| F1__remendo | 0.362 | 0.028 |
| mAP50_95__buraco | 0.236 | 0.044 |
| mAP50_95__rachadura | 0.087 | 0.048 |
| mAP50_95__remendo | 0.189 | 0.028 |
| mAP50__buraco | 0.501 | 0.062 |
| mAP50__rachadura | 0.216 | 0.108 |
| mAP50__remendo | 0.326 | 0.050 |

Mediana da melhor época: **74** (épocas do treino definitivo, E8).
Limiar de confiança que maximiza o F1 (mediana): **0.22122122122122123** (use no teste com `final_test.py --conf cv`).

Observação: o fold 0 foi usado em todas as decisões (modelo, tamanho, resolução, CLAHE, aumento e Optuna); os folds 1 e 2 não participaram de nenhuma escolha. Se a média dos folds 1 e 2 ficar bem abaixo do fold 0, as escolhas se ajustaram demais ao fold 0.
