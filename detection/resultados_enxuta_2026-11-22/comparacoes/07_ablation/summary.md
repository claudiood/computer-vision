# Comparação — 07_ablation

Métrica de seleção: **mAP50_95** (validação). Ruído (desvio entre seeds do baseline): **0.0037**

## 07_ablation

Referência: `full`.

| variante | n | mAP50_95 | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | F1 buraco | F1 rachadura | F1 remendo | treino (min) | inferência (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| full | 3 | 0.2212 | 0.0147 | +0.0000 | não | 0.392 | 0.411 | 0.322 | 0.406 | 0.248 | 0.579 | 5.9 | 10.8 |
| minus_model | 3 | 0.2212 | 0.0147 | +0.0000 | não | 0.392 | 0.411 | 0.322 | 0.406 | 0.248 | 0.579 | 5.8 | 7.8 |
| minus_imgsz | 3 | 0.1804 | 0.0104 | -0.0408 | pior | 0.343 | 0.386 | 0.364 | 0.400 | 0.203 | 0.554 | 4.3 | 4.9 |

Melhor: **full** (mAP50_95 = 0.2212, Δ = +0.0000; supera o ruído: não).

