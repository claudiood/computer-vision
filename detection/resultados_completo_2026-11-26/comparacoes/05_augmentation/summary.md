# Comparação — 05_augmentation

Métrica de seleção: **mAP50_95** (validação). Δ ref = diferença contra a referência, fold a fold. Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = menor que a margem (ruído).

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

