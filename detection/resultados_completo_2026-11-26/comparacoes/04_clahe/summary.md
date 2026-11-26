# Comparação — 04_clahe

Métrica de seleção: **mAP50_95** (validação). Δ ref = diferença contra a referência, fold a fold. Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = menor que a margem (ruído).

## 04_clahe

Referência: `sem CLAHE (E1-E3)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| sem CLAHE (E1-E3) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| clahe1 | 0 | 0.1772 | nan | -0.0046 | empate | f0: -0.005 | 0.333 | 0.377 | 0.531 | 0.251 | 0.347 | 3.6 |
| clahe2 | 0 | 0.1600 | nan | -0.0218 | pior | f0: -0.022 | 0.312 | 0.338 | 0.464 | 0.204 | 0.345 | 3.0 |

Maior valor: **sem CLAHE (E1-E3)** (mAP50_95 = 0.1818, Δ = +0.0000, veredito: referência). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

