# Comparação — 02_models

Métrica de seleção: **mAP50_95** (validação). Δ ref = diferença contra a referência, fold a fold. Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = menor que a margem (ruído).

## 02_models

Referência: `n (E1)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| s | 0 | 0.1878 | nan | +0.0060 | empate | f0: +0.006 | 0.379 | 0.403 | 0.510 | 0.280 | 0.418 | 4.4 |
| n (E1) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| m | 0 | 0.1499 | nan | -0.0319 | pior | f0: -0.032 | 0.304 | 0.346 | 0.483 | 0.241 | 0.314 | 6.2 |

Maior valor: **s** (mAP50_95 = 0.1878, Δ = +0.0060, veredito: empate). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

