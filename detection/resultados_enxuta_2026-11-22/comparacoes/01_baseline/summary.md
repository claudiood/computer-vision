# Comparação — 01_baseline

Métrica de seleção: **mAP50_95** (validação). Ruído (desvio entre seeds do baseline): **0.0037**

## 01_baseline

Referência: `yolo26n_640`.

| variante | n | mAP50_95 | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | F1 buraco | F1 rachadura | F1 remendo | treino (min) | inferência (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n_640 | 3 | 0.1399 | 0.0037 | +0.0000 | não | 0.308 | 0.361 | 0.314 | 0.382 | 0.190 | 0.509 | 5.1 | 3.3 |

Melhor: **yolo26n_640** (mAP50_95 = 0.1399, Δ = +0.0000; supera o ruído: não).

