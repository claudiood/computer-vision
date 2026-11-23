# Comparação — 02_models

Métrica de seleção: **mAP50_95** (validação). Ruído (desvio entre seeds do baseline): **0.0037**

## 02_models

Referência: `yolo26n`.

| variante | n | mAP50_95 | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | F1 buraco | F1 rachadura | F1 remendo | treino (min) | inferência (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 1 | 0.1390 | nan | +0.0000 | não | 0.300 | 0.356 | 0.342 | 0.420 | 0.250 | 0.398 | 5.6 | 3.5 |
| yolo11n | 1 | 0.1093 | nan | -0.0296 | pior | 0.264 | 0.337 | 0.337 | 0.432 | 0.172 | 0.406 | 4.6 | 3.8 |
| yolo26s | 1 | 0.1063 | nan | -0.0326 | pior | 0.241 | 0.259 | 0.260 | 0.245 | 0.122 | 0.409 | 5.7 | 7.3 |

Melhor: **yolo26n** (mAP50_95 = 0.1390, Δ = +0.0000; supera o ruído: não).

