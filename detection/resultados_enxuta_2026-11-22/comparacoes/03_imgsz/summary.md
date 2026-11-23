# Comparação — 03_imgsz

Métrica de seleção: **mAP50_95** (validação). Ruído (desvio entre seeds do baseline): **0.0037**

## 03_imgsz

Referência: `imgsz_640`.

| variante | n | mAP50_95 | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | F1 buraco | F1 rachadura | F1 remendo | treino (min) | inferência (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| imgsz_960 | 1 | 0.1636 | nan | +0.0246 | sim | 0.304 | 0.381 | 0.317 | 0.455 | 0.212 | 0.478 | 6.9 | 19.7 |
| imgsz_640 | 1 | 0.1390 | nan | +0.0000 | não | 0.300 | 0.356 | 0.342 | 0.420 | 0.250 | 0.398 | 5.9 | 4.0 |

Melhor: **imgsz_960** (mAP50_95 = 0.1636, Δ = +0.0246; supera o ruído: sim).

