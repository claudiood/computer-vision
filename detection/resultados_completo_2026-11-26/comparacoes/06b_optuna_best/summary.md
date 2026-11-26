# Comparação — 06b_optuna_best

Métrica de seleção: **mAP50_95** (validação). Δ ref = diferença contra a referência, fold a fold. Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = menor que a margem (ruído).

## 06b_optuna_best

Referência: `vigente (E1-E5)`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| best_trial | 0 | 0.1865 | nan | +0.0048 | empate | f0: +0.005 | 0.373 | 0.416 | 0.515 | 0.277 | 0.455 | 3.0 |
| vigente (E1-E5) | 0 | 0.1818 | nan | +0.0000 | referência |  | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |

Maior valor: **best_trial** (mAP50_95 = 0.1865, Δ = +0.0048, veredito: empate). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

