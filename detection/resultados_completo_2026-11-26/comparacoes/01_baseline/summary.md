# Comparação — 01_baseline

Métrica de seleção: **mAP50_95** (validação). Δ ref = diferença contra a referência, fold a fold. Veredito: **melhor**/**pior** = diferença maior que a margem de adoção (0,015); **empate** = menor que a margem (ruído).

## 01_baseline

Referência: `yolo11n`.

| variante | folds | mAP50_95 | ± | Δ ref | veredito | Δ por fold | mAP50 | F1 | F1 buraco | F1 rachadura | F1 remendo | treino (min) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 0 | 0.1818 | nan | +0.0363 | melhor | f0: +0.036 | 0.353 | 0.372 | 0.514 | 0.227 | 0.375 | 2.6 |
| yolo11n | 0 | 0.1454 | nan | +0.0000 | referência |  | 0.310 | 0.358 | 0.488 | 0.228 | 0.357 | 2.5 |

Maior valor: **yolo26n** (mAP50_95 = 0.1818, Δ = +0.0363, veredito: melhor). Os próximos experimentos só adotam uma variante nova se ela superar a configuração vigente por mais que a margem; num empate, fica a vigente.

