# Comparação — 04_augmentation

Métrica de seleção: **mAP50_95** (validação). Ruído (desvio entre seeds do baseline): **0.0037**

## 04_augmentation

Referência: `control`.

| variante | n | mAP50_95 | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | F1 buraco | F1 rachadura | F1 remendo | treino (min) | inferência (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| control | 1 | 0.1636 | nan | +0.0000 | não | 0.304 | 0.381 | 0.317 | 0.455 | 0.212 | 0.478 | 6.8 | 14.4 |
| light_strong | 1 | 0.1533 | nan | -0.0102 | pior | 0.339 | 0.375 | 0.295 | 0.410 | 0.247 | 0.468 | 6.0 | 10.9 |
| scale_strong | 1 | 0.1245 | nan | -0.0391 | pior | 0.297 | 0.360 | 0.252 | 0.415 | 0.213 | 0.453 | 6.6 | 10.3 |
| no_aug | 1 | 0.0442 | nan | -0.1194 | pior | 0.116 | 0.155 | 0.106 | 0.283 | 0.183 | 0.000 | 3.8 | 6.3 |

Melhor: **control** (mAP50_95 = 0.1636, Δ = +0.0000; supera o ruído: não).

