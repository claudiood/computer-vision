# Roteiro de treinamento — Detecção de danos em pavimentos com YOLO (pavement_v3)

Este roteiro cobre o pipeline para treinar um detector de três classes de dano em pavimento: **rachadura** (crack), **remendo** (patch) e **buraco** (pothole). Os dados de origem ficam em `dataset/detection` e o código na pasta `detection/`.

Esta é a versão para a **nova rotulação** (`pavement_v3`). A rodada anterior (`pavement_v2`, resultados em `resultados_enxuta_2026-11-22/`) foi descartada: as 32 imagens sem anotação entravam como fundo mesmo tendo danos visíveis, o critério de anotação mudava entre as pessoas e a divisão não separava fotos da mesma cena. Os números daquela rodada não são comparáveis com os desta.

Orçamento: **cerca de 5 horas** numa A6000 com 6 vCPUs (seção 6).

---

## 1. Antes de treinar: dados

### 1.1 Guia de rotulação (todos seguem o mesmo)

| Classe | Como anotar | Não é esta classe |
|---|---|---|
| rachadura (0) | Uma caixa por **trecho contínuo**. Rachadura longa ou diagonal vira várias caixas seguidas, nunca uma caixa da rua inteira. | Faixa pintada, junta de dilatação, desgaste superficial, sombra |
| remendo (1) | Uma caixa cobrindo a área do reparo | Tampa de bueiro, meio-fio, grade |
| buraco (2) | Depressão com perda de material; caixa justa na borda. Buraco dentro de remendo: anote os dois. | Poça sem buraco, sombra |

Regras gerais:

- Anote **todos** os danos visíveis da imagem. Dano sem caixa vira erro na avaliação.
- Danos muito pequenos ou ilegíveis ficam de fora, do mesmo jeito para todo mundo.
- **Imagem sem `.txt` fica fora do dataset** (`unlabeled_policy: exclude`). Uma imagem só entra como fundo se alguém conferiu que ela não tem dano nenhum; nesse caso, liste-a em `background_images` no `configs/dataset.yaml`.
- Revisão cruzada: cada imagem é conferida por uma segunda pessoa. Vale começar com umas 10 imagens que todos anotam e comparar, para calibrar o grupo.
- Ao exportar do Make Sense, confira se a ordem do `labels.txt` continua `rachadura, remendo, buraco`.

### 1.2 O que muda no preparo (`configs/dataset.yaml`)

| Item | Valor | Por quê |
|---|---|---|
| `unlabeled_policy` | `exclude` | Fundo com dano ensina o modelo a ignorar o dano e conta acerto como falso positivo |
| `grouping` | `hash_threshold: 70`, `time_window_s: 30`, `seq_window: 2` | Fotos da mesma cena (mesmo celular, segundos de diferença) ficam sempre na mesma partição. Sem isso, a validação e o teste avaliam cenas já vistas no treino |
| `prepare.name` | `pavement_v3` | Versão nova: divisão e manifesto novos |
| `prepare.max_side` | `1280` | Igual ao maior `imgsz` testado. Imagens menores são mais rápidas de ler |

A divisão continua: teste fixo de 15% (usado **uma vez**, no fim) e 5 folds sobre o resto, estratificados por classe e por anotador.

---

## 2. Estrutura do projeto

```
detection/
├── TRAINING_ROADMAP.md          este documento
├── requirements.txt
├── configs/
│   ├── dataset.yaml             caminhos, classes, política de imagens sem rótulo, divisão, agrupamento
│   ├── base_train.yaml          configuração-base de todo treino
│   ├── optuna.yaml              E5: espaço de busca e configuração do estudo
│   └── experiments/
│       01_baseline.yaml  02_models.yaml  03_imgsz.yaml  04_clahe.yaml
│       06_ablation.yaml  07_cv.yaml      08_final.yaml
├── pavedet/                     funções reutilizáveis
│   ├── config.py                YAML, herança (extends), overrides (--set chave=valor)
│   ├── io_utils.py              imagens (HEIC/EXIF), rótulos YOLO, limpeza, dHash
│   ├── dataset.py               leitura do dataset de origem e do anotador
│   ├── splitting.py             grupos de mesma cena e divisão estratificada por grupo
│   ├── preprocess.py            CLAHE
│   ├── training.py              monta a configuração (herança), treina, valida e registra; oversampling
│   ├── metrics.py               P/R/F1/mAP por classe e ponto de operação
│   ├── evaluation.py            avaliação completa e tempo de inferência
│   ├── tracking.py              registro (run_meta.json, registry.jsonl), melhor variante
│   ├── plots.py                 gráficos
│   └── env.py                   device, versões, memória
├── scripts/
│   00_check_env.py  01_audit_dataset.py  02_prepare_dataset.py  03_validate_dataset.py
│   run_experiment.py  tune_optuna.py  compare.py  cv_report.py
│   evaluate.py  final_test.py  predict.py  train.py (treino avulso)
├── tests/                       dataset sintético e smoke_test.sh
├── splits/pavement_v3.csv       manifesto da divisão (VERSIONADO: a mesma divisão para todo o grupo)
├── data/                        datasets preparados (gerado; fora do git)
└── results/                     execuções, comparações, Optuna, CV e teste (fora do git)
```

### 2.1 Como a configuração de cada treino é montada

Cada execução junta as camadas abaixo, nesta ordem (a de baixo vence):

1. `base_train.yaml`
2. A melhor configuração entre os experimentos listados em `inherit_best` (maior média de mAP50-95 nos folds). Exemplo: o E4 herda o melhor entre E1, E2 e E3.
3. `overrides` do experimento
4. A variante que está sendo testada
5. `--set chave=valor` na linha de comando

Com isso, cada experimento parte automaticamente do vencedor dos anteriores.

### 2.2 Referências sem treinar de novo (`include_best`)

A configuração que já venceu não é treinada de novo. A chave `include_best` do YAML traz as execuções dela para a comparação. Exemplo: no E2, o yolo26n do E1 aparece como `yolo26n (E1)`; no E3, o 960 aparece como `imgsz 960 (E1/E2)`.

### 2.3 Variantes repetidas (`skip_duplicates`)

Na ablação, uma variante que ficaria igual à `full` é pulada. Exemplo: se o CLAHE não foi adotado, `minus_clahe` é idêntica à `full` e não roda.

### 2.4 O que é registrado

Cada execução tem a pasta `results/experiments/<experimento>/<variante>/s<seed>_f<fold>/` com os arquivos da Ultralytics (`weights/`, `results.csv`, curvas), o `run_meta.json` (configuração completa, ambiente, tempos, melhor época, métricas gerais e por classe) e `val_best/`. Cada execução também acrescenta uma linha em `results/registry.jsonl`, que só cresce.

Uma execução já concluída é pulada (dá para interromper e continuar). `--force` roda de novo, preservando a pasta antiga.

---

## 3. Como decidir: comparação pareada por fold

Com cerca de 30 imagens por fold de validação, um único número engana: na rodada anterior, a mesma configuração variou ±0,015 de mAP50-95 entre seeds, e o Optuna escolhido no fold 0 caiu de 0,22 na validação para 0,09 no teste.

Por isso, **todas as decisões usam os mesmos 3 folds (0, 1, 2)**, e o `compare.py` compara cada variante com a referência **fold contra o mesmo fold**:

| Veredito | Quando |
|---|---|
| **melhor** | ganhou nos 3 folds |
| **pior** | perdeu nos 3 folds |
| **empate** | o sinal muda entre os folds: a diferença não é confiável |

A herança (`inherit_best`) usa a maior média. Se o veredito for **empate**, prefira a opção mais simples ou barata (modelo menor, imgsz menor, sem CLAHE) e passe-a com `--set` nos experimentos seguintes. Registre a decisão: "não muda nada" também é resultado.

---

## 4. Pipeline

| # | Etapa | Variantes | Treinos |
|---|---|---|---|
| E0 | Ambiente, auditoria, preparo, validação dos dados | — | — |
| E1 | Baseline | yolo26n, 960, padrões | 3 |
| E2 | Modelos | yolo26s, yolo11n (yolo26n vem do E1) | 6 |
| E3 | Resolução | 1280 no melhor modelo (960 vem de antes) | 3 |
| E4 | CLAHE | clipLimit 1.0 e 2.0 | 6 |
| E5 | Optuna | hiperparâmetros + aumento de dados + oversampling, fold 0 | 12 trials |
| E6 | Ablação | full, minus_imgsz, minus_model, minus_clahe (só as adotadas); "sem Optuna" vem do E1–E4 | 3 a 12 |
| E7 | Validação cruzada | vencedora nos folds 3 e 4 (0–2 já existem) | 2 |
| E8 | Final + teste | treino+validação, épocas da CV, 3 seeds; teste uma vez | 3 |

Cerca de **38 a 47 treinos**. Depois de cada experimento: `python scripts/compare.py --experiments <nome>`.

### E0 — Ambiente e dados

```bash
cd detection
pip install -r requirements.txt
python scripts/00_check_env.py --smoke                 # versões, device e 1 época no COCO8
bash tests/smoke_test.sh --quick                       # confere preparo e dry-runs (sem treinos)
python scripts/01_audit_dataset.py                     # auditoria da nova rotulação: results/audit/
python scripts/02_prepare_dataset.py --resplit         # divisão nova → splits/pavement_v3.csv
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 1.0   # → data/pavement_v3_clahe1
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 2.0   # → data/pavement_v3_clahe2
python scripts/03_validate_dataset.py --ultralytics
```

Confira na auditoria: nenhuma imagem com dano ficou sem `.txt`, o número de caixas de rachadura por imagem ficou parecido entre anotadores e o `split_report.md` mostra **0 grupos atravessando partições**. Depois, versione o `splits/pavement_v3.csv` no git, para todos usarem a mesma divisão.

### E1 — Baseline

| | |
|---|---|
| Configuração | `yolo26n.pt`, `imgsz 960`, padrões da Ultralytics, `epochs 100`, `patience 25`, `batch 8` |
| Por quê | Ponto de partida e referência dos experimentos seguintes. O 960 já se mostrou melhor que 640 na rodada anterior. |

```bash
python scripts/run_experiment.py 01_baseline
```

### E2 — Modelos

| | |
|---|---|
| Variantes | `yolo26s`, `yolo11n` (o `yolo26n` entra como referência, vindo do E1) |
| Por quê | Um modelo maior (s) e outra geração (11n). Modelos m/l ficam de fora: com ~120 imagens de treino tendem a decorar o treino. |

```bash
python scripts/run_experiment.py 02_models
python scripts/compare.py --experiments 02_models
```

### E3 — Resolução

| | |
|---|---|
| Variante | `imgsz_1280` no melhor modelo do E1/E2 (o 960 entra como referência) |
| Por quê | Rachaduras são finas e somem em resoluções menores. Foi o único ganho consistente da rodada anterior (640 → 960). |
| Observação | Se o E2 der empate entre dois modelos, rode também o 1280 no outro: `--set model=<modelo>.pt` |

```bash
python scripts/run_experiment.py 03_imgsz
python scripts/compare.py --experiments 03_imgsz
```

### E4 — CLAHE

| | |
|---|---|
| Variantes | `clahe1` (clipLimit 1.0) e `clahe2` (clipLimit 2.0); "sem CLAHE" entra como referência |
| Por quê | Realça contraste local (sombra, sol forte). É pré-processamento: aplicado igual no treino, validação, teste e uso (`predict.py` reaplica sozinho). |

```bash
python scripts/run_experiment.py 04_clahe
python scripts/compare.py --experiments 04_clahe
```

### E5 — Optuna (hiperparâmetros e aumento de dados)

| | |
|---|---|
| Base | O melhor entre E1–E4 |
| Espaço de busca | otimizador, lr0, box, cls; scale, mosaic, close_mosaic, translate, hsv_v, degrees, mixup; oversampling do remendo (×1, ×2, ×3) |
| Configuração | 12 trials, 50 épocas e paciência 15 por trial, poda pela mediana, só no fold 0 |
| Por quê | Substitui os experimentos manuais de aumento de dados e desbalanceamento. O espaço foi reduzido aos parâmetros que pesaram na rodada anterior (cls, lr0, close_mosaic, hsv_v, mosaic, degrees) mais os que faltavam testar. |

```bash
python scripts/tune_optuna.py
python scripts/tune_optuna.py --report        # importâncias e histórico em results/optuna/hparams_v3/
```

O melhor trial **não é adotado direto**: ele foi escolhido olhando só o fold 0, com treinos curtos. Quem decide é a ablação.

### E6 — Ablação

| | |
|---|---|
| Variantes | `full` (melhor trial do Optuna, com 100 épocas, folds 0–2), `minus_imgsz`, `minus_model`, `minus_clahe`. A configuração **sem Optuna** entra como referência, vinda do E1–E4. |
| Variantes puladas | As que ficariam iguais à `full` (componente não adotado) |
| Por quê | Mostra quanto cada decisão contribui e confirma se o Optuna ajudou em folds que ele não viu. Se `full` não superar a "sem Optuna", o E7 e o E8 herdam automaticamente a sem Optuna. |

```bash
python scripts/run_experiment.py 06_ablation
python scripts/compare.py --experiments 06_ablation
```

### E7 — Validação cruzada

| | |
|---|---|
| O que faz | Treina a configuração vencedora (o melhor entre E1, E2, E3, E4 e E6) nos folds 3 e 4. O `cv_report.py` junta com os folds 0–2 já treinados. |
| Saídas | `results/cv/cv_report.md`: mAP50-95, mAP50, P, R, F1 em **média ± desvio de 5 folds** (o número principal do relatório), por classe; a mediana da melhor época (usada no E8) e o limiar de confiança que maximiza o F1. |

```bash
python scripts/run_experiment.py 07_cv
python scripts/cv_report.py
```

### E8 — Treino definitivo e teste

| | |
|---|---|
| Configuração | A vencedora, treinada em **treino + validação** (todas as imagens menos o teste), com o número de épocas fixo na mediana da melhor época da CV e sem early stopping. 3 seeds. |
| Por quê | Cerca de 20% a mais de dados, o que pesa num dataset deste tamanho. O número de épocas vem da CV porque não sobra validação independente. Vale o modelo da última época (`last.pt`). |
| Teste | Usado **uma vez**. Resultado = média ± desvio das 3 seeds. O limiar do ponto de operação vem da CV (`--conf cv`), não do teste. |

```bash
python scripts/run_experiment.py 08_final
python scripts/final_test.py --experiment 08_final --confirm --conf cv
```

O `final_test.py` grava `results/final/TEST_LOCK.json` e recusa rodar de novo sem `--rerun` (que fica registrado no relatório). O modelo entregue é o da seed 0, com o caminho no `test_report.md`.

### Uso do modelo em imagens novas

```bash
python scripts/predict.py --run results/experiments/08_final/final_trainval/s0_ftrainval --source pasta_ou_imagem --conf <limiar da CV>
```

---

## 5. Métricas

| Métrica | Uso |
|---|---|
| **mAP50-95** | Métrica de seleção. Média do AP em IoU de 0,50 a 0,95: cobra caixas precisas. |
| mAP50 | AP com IoU ≥ 0,5: o modelo achou o dano mais ou menos no lugar certo |
| P, R, F1 | No limiar que maximiza o F1 (`best_conf`) |
| P_op, R_op, F1_op | No limiar fixo do ponto de operação, com matriz de confusão própria. É o que o usuário vê com `predict`. |
| Por classe | Todas as anteriores por classe. Rachadura costuma ser a mais baixa (extensão da caixa ambígua). |

Referência: em datasets públicos grandes de pavimento (milhares de imagens), modelos bem treinados ficam em torno de 0,5–0,65 de mAP50. Para cerca de 170 imagens bem anotadas, 0,35–0,45 de mAP50 no teste seria um bom resultado.

---

## 6. Tempo: caber em 5 horas

### 6.1 Já configurado

| Ajuste | Onde | Efeito |
|---|---|---|
| `cache: disk` | `base_train.yaml` | Guarda as imagens decodificadas: a CPU não redecodifica o JPEG a cada época. Na rodada anterior a GPU usava 2,4 de 50 GB; o gargalo era a CPU. |
| `max_side: 1280` | `dataset.yaml` | Imagens menores, leitura mais rápida, sem perda (o maior imgsz é 1280) |
| `workers: 6` | `base_train.yaml` | Usa todos os vCPUs |
| `epochs 100`, `patience 25` | `base_train.yaml` | O early stopping corta o desnecessário |
| Optuna: 12 trials de 50 épocas, poda | `optuna.yaml` | |
| Referências reaproveitadas, ablação pula variantes repetidas, CV só treina 2 folds | YAMLs | Nenhum treino repetido |

### 6.2 Estimativa (tempos da rodada anterior)

| Etapa | Treinos | Tempo |
|---|---|---|
| E1 | 3 | 15 min |
| E2 | 6 | 35 min |
| E3 | 3 | 30 min |
| E4 | 6 | 35–55 min |
| E5 | 12 | 50 min |
| E6 | 3–12 | 20–80 min |
| E7 | 2 | 15 min |
| E8 + teste | 3 | 20–30 min |
| Comparações e relatórios | — | 20 min |
| **Total** | **~38–47** | **~4h–5h30** |

O tempo maior vale se o 1280 vencer (as etapas seguintes ficam mais lentas). Com o cache, deve sobrar folga. Rode um treino curto antes (`python scripts/train.py --name teste --set train.epochs=10`) para medir o tempo por época na máquina.

### 6.3 Dois treinos ao mesmo tempo (opcional)

A GPU comporta dois treinos. Em dois terminais:

```bash
python scripts/run_experiment.py 02_models --variants yolo26s
python scripts/run_experiment.py 02_models --variants yolo11n
```

O mesmo vale para `04_clahe` (`clahe1` e `clahe2`) e para o Optuna (o mesmo comando em dois terminais divide os trials). Com só 6 vCPUs o ganho é limitado; teste antes.

### 6.4 Se estourar o tempo, corte nesta ordem

1. Optuna de 12 para 8 trials: `python scripts/tune_optuna.py --n-trials 8`
2. Ablação só com `full` e `minus_imgsz`: `--variants full minus_imgsz`
3. Final com 1 seed: `python scripts/run_experiment.py 08_final --seeds 0` (o teste fica sem desvio)

Não corte os 3 folds do E1 ao E4: é o que garante que as escolhas não são sorte.

---

## 7. Ordem resumida

| # | Comando |
|---|---|
| E0 | `00_check_env.py --smoke` → `01_audit_dataset.py` → `02_prepare_dataset.py --resplit` → `02_prepare_dataset.py --preprocess clahe --clahe-clip 1.0` e `2.0` → `03_validate_dataset.py --ultralytics` |
| E1 | `run_experiment.py 01_baseline` |
| E2 | `run_experiment.py 02_models` → `compare.py --experiments 02_models` |
| E3 | `run_experiment.py 03_imgsz` → `compare.py --experiments 03_imgsz` |
| E4 | `run_experiment.py 04_clahe` → `compare.py --experiments 04_clahe` |
| E5 | `tune_optuna.py` → `tune_optuna.py --report` |
| E6 | `run_experiment.py 06_ablation` → `compare.py --experiments 06_ablation` |
| E7 | `run_experiment.py 07_cv` → `cv_report.py` |
| E8 | `run_experiment.py 08_final` → `final_test.py --experiment 08_final --confirm --conf cv` |
| Relatório | `compare.py --experiments 01_baseline 02_models 03_imgsz 04_clahe 06_ablation --out resumo` |

---

## 8. Comandos úteis

| Comando | Para quê |
|---|---|
| `python scripts/run_experiment.py --list` | Lista os experimentos |
| `python scripts/run_experiment.py 03_imgsz --dry-run` | Mostra a configuração herdada sem treinar |
| `python scripts/run_experiment.py 02_models --variants yolo26s` | Só uma variante |
| `python scripts/run_experiment.py 04_clahe --set train.batch=4` | Muda um valor em todas as variantes |
| `python scripts/train.py --name teste --set train.epochs=10` | Treino avulso, também registrado |
| `python scripts/evaluate.py --best-of 06_ablation` | Avaliação detalhada na validação: matrizes, curvas, piores imagens, velocidade |
| `bash tests/smoke_test.sh` | Teste do pipeline inteiro num dataset sintético, na CPU (sem tocar em `data/` e `results/`) |

O `evaluate.py` recusa o teste. O teste só é usado pelo `final_test.py`.

---

## 9. Limitações a relatar

- **Dataset pequeno:** cerca de 170 imagens. Os folds têm ~30 imagens e o teste ~25; as métricas variam bastante, por isso a comparação pareada e a CV de 5 folds.
- **Fold 0 usado pelo Optuna:** a ablação treina nos folds 0–2, então a `full` tem uma pequena vantagem no fold 0. Os folds 3 e 4 da CV não participaram de nenhuma escolha.
- **Rachadura:** mesmo com o guia, a extensão de uma rachadura é ambígua; espera-se mAP50-95 mais baixo nessa classe.
- **Agrupamento por cena:** depende de EXIF (data e câmera) e da numeração das fotos. Fotos sem EXIF podem escapar do agrupamento; confira o `split_report.md`.

---

## 10. Pendências antes do primeiro treino

1. Terminar a nova rotulação seguindo a seção 1.1.
2. Apagar os YAMLs da rodada anterior que não fazem mais parte do pipeline: `configs/experiments/04_augmentation.yaml`, `05_imbalance.yaml`, `06b_optuna_confirm.yaml` e `07_ablation.yaml` (`git rm`).
3. Apagar ou mover a pasta `results/` antiga da máquina de treino, se existir.
4. Manter a mesma máquina e as mesmas versões em todos os experimentos.
5. Rodar `bash tests/smoke_test.sh --quick` uma vez na máquina de treino.
