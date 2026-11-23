# Comparação — 05_imbalance_06b_optuna_confirm

Métrica de seleção: **mAP50_95** (validação). Ruído (desvio entre seeds do baseline): **0.0037**

## 05_imbalance

Referência: `control`.

| variante | n | mAP50_95 | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | F1 buraco | F1 rachadura | F1 remendo | treino (min) | inferência (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| patch_x2 | 1 | 0.1689 | nan | +0.0053 | sim | 0.348 | 0.381 | 0.397 | 0.380 | 0.256 | 0.507 | 8.7 | 22.3 |
| control | 1 | 0.1636 | nan | +0.0000 | não | 0.304 | 0.381 | 0.317 | 0.455 | 0.212 | 0.478 | 6.8 | 13.9 |
| rfs_t05 | 1 | 0.1595 | nan | -0.0041 | pior | 0.344 | 0.387 | 0.354 | 0.389 | 0.250 | 0.522 | 5.4 | 7.2 |

Melhor: **patch_x2** (mAP50_95 = 0.1689, Δ = +0.0053; supera o ruído: sim).

## 06b_optuna_confirm

Referência: `trial_0008`.

| variante | n | mAP50_95 | ± | Δ ref | supera ruído | mAP50 | F1 | F1_op | F1 buraco | F1 rachadura | F1 remendo | treino (min) | inferência (ms) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| trial_0010 | 2 | 0.2294 | 0.0052 | +0.0496 | sim | 0.402 | 0.432 | 0.313 | 0.416 | 0.284 | 0.596 | 5.0 | 11.7 |
| trial_0009 | 2 | 0.1946 | 0.0002 | +0.0149 | sim | 0.369 | 0.417 | 0.276 | 0.397 | 0.284 | 0.570 | 5.7 | 9.0 |
| trial_0008 | 2 | 0.1798 | 0.0218 | +0.0000 | não | 0.365 | 0.430 | 0.431 | 0.457 | 0.164 | 0.669 | 5.1 | 9.1 |

Melhor: **trial_0010** (mAP50_95 = 0.2294, Δ = +0.0496; supera o ruído: sim).

