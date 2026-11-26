# Comparação — 03_imgsz

Métrica de seleção: **mAP50_95** (validação). Δ ref = diferença contra a referência, fold a fold. Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = menor que a margem (ruído).

## 03_imgsz

Referência: `imgsz 960 (E1/E2)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| imgsz 960 (E1/E2) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| imgsz_1600 | 0 | 0.1617 | nan | -0.0200 | pior | f0: -0.020 | 0.335 | 0.384 | 0.526 | 0.238 | 0.389 | 5.1 |
| imgsz_1280 | 0 | 0.1603 | nan | -0.0214 | pior | f0: -0.021 | 0.341 | 0.366 | 0.482 | 0.253 | 0.364 | 2.7 |

Maior valor: **imgsz 960 (E1/E2)** (mAP50_95 = 0.1818, Δ = +0.0000, veredito: referência). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

