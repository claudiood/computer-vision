# Comparação — resumo

Métrica de seleção: **mAP50_95** (validação). Δ ref = diferença contra a referência, fold a fold. Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = menor que a margem (ruído).

## 01_baseline

Referência: `yolo11n`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 0 | 0.1818 | nan | +0.0363 | melhor | f0: +0.036 | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| yolo11n | 0 | 0.1454 | nan | +0.0000 | referência |  | 0.310 | 0.358 | 0.488 | 0.228 | 0.357 | 2.5 |

Maior valor: **yolo26n** (mAP50_95 = 0.1818, Δ = +0.0363, veredito: melhor). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

## 02_models

Referência: `n (E1)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s | 0 | 0.1878 | nan | +0.0060 | empate | f0: +0.006 | 0.379 | 0.403 | 0.510 | 0.280 | 0.418 | 4.4 |
| n (E1) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| m | 0 | 0.1499 | nan | -0.0319 | pior | f0: -0.032 | 0.304 | 0.346 | 0.483 | 0.241 | 0.314 | 6.2 |

Maior valor: **s** (mAP50_95 = 0.1878, Δ = +0.0060, veredito: empate). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

## 03_imgsz

Referência: `imgsz 960 (E1/E2)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| imgsz 960 (E1/E2) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| imgsz_1600 | 0 | 0.1617 | nan | -0.0200 | pior | f0: -0.020 | 0.335 | 0.384 | 0.526 | 0.238 | 0.389 | 5.1 |
| imgsz_1280 | 0 | 0.1603 | nan | -0.0214 | pior | f0: -0.021 | 0.341 | 0.366 | 0.482 | 0.253 | 0.364 | 2.7 |

Maior valor: **imgsz 960 (E1/E2)** (mAP50_95 = 0.1818, Δ = +0.0000, veredito: referência). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

## 04_clahe

Referência: `sem CLAHE (E1-E3)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sem CLAHE (E1-E3) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| clahe1 | 0 | 0.1772 | nan | -0.0046 | empate | f0: -0.005 | 0.333 | 0.377 | 0.531 | 0.251 | 0.347 | 3.6 |
| clahe2 | 0 | 0.1600 | nan | -0.0218 | pior | f0: -0.022 | 0.312 | 0.338 | 0.464 | 0.204 | 0.345 | 3.0 |

Maior valor: **sem CLAHE (E1-E3)** (mAP50_95 = 0.1818, Δ = +0.0000, veredito: referência). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

## 05_augmentation

Referência: `padrão (E1-E4)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| luz_forte | 0 | 0.1909 | nan | +0.0092 | empate | f0: +0.009 | 0.362 | 0.408 | 0.517 | 0.263 | 0.444 | 3.7 |
| padrão (E1-E4) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| close_mosaic_20 | 0 | 0.1818 | nan | +0.0000 | empate | f0: +0.000 | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.4 |
| mixup | 0 | 0.1721 | nan | -0.0096 | empate | f0: -0.010 | 0.323 | 0.350 | 0.457 | 0.231 | 0.363 | 3.6 |
| geometria | 0 | 0.1556 | nan | -0.0261 | pior | f0: -0.026 | 0.342 | 0.384 | 0.581 | 0.264 | 0.306 | 3.3 |

Maior valor: **luz_forte** (mAP50_95 = 0.1909, Δ = +0.0092, veredito: empate). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

## 06b_optuna_best

Referência: `vigente (E1-E5)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_trial | 0 | 0.1865 | nan | +0.0048 | empate | f0: +0.005 | 0.373 | 0.416 | 0.515 | 0.277 | 0.455 | 3.0 |
| vigente (E1-E5) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |

Maior valor: **best_trial** (mAP50_95 = 0.1865, Δ = +0.0048, veredito: empate). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

