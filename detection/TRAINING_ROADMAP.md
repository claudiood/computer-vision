# Roteiro de treinamento — Detecção de danos em pavimentos com YOLO (pavement_v3)

Este roteiro cobre o pipeline para treinar um detector de três classes de dano em pavimento: **rachadura** (crack), **remendo** (patch) e **buraco** (pothole). Os dados de origem ficam em `dataset/detection/images` e `dataset/detection/labels` (com o `labels.txt` do Make Sense), e o código na pasta `detection/`.

Esta é a versão para a **nova rotulação** (`pavement_v3`). A rodada anterior (`pavement_v2`, resultados em `resultados_enxuta_2026-11-22/`) foi descartada: várias imagens sem anotação tinham danos visíveis e o critério de anotação mudava entre as pessoas. Os números daquela rodada não são comparáveis com os desta.

Orçamento: **cerca de 3 a 4 horas** numa A6000 com 6 vCPUs, se vencer um modelo n ou s a 960/1280 (seção 6).

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
- **Imagem sem `.txt` entra como fundo** (imagem sem dano, `unlabeled_policy: background`). Por isso, toda imagem com dano visível precisa ter `.txt`: uma foto com dano e sem anotação ensina o modelo a ignorar o dano, e um acerto nela conta como falso positivo. Se não der para anotar, liste-a em `exclude` no `configs/dataset.yaml`.
- Ao exportar do Make Sense, confira se a ordem do `labels.txt` continua `rachadura, remendo, buraco`.

### 1.2 Preparo (`configs/dataset.yaml`)

| Item | Valor | Por quê |
|---|---|---|
| `source` | `../dataset/detection/images` e `labels` | Pasta plana com as imagens e os `.txt` |
| `unlabeled_policy` | `background` | Decisão do grupo: imagens sem `.txt` entram como imagens sem dano |
| `grouping` | tudo em `0` (desligado) | Decisão do grupo: divisão por imagem, sem agrupar fotos da mesma cena |
| `prepare.name` | `pavement_v3` | Versão nova: divisão e manifesto novos |
| `prepare.max_side` | `1600` | Igual ao maior `imgsz` testado: o 1600 não fica só ampliado, e as imagens são menores que as fotos de 12 MP |
| `prepare.folds` | `3` | O fold 0 decide tudo; os 3 juntos formam a validação cruzada do final |

Divisão: teste fixo de 15% (usado **uma vez**, no fim) e 3 folds sobre o resto, estratificados por classe e por anotador. Cada fold de validação tem cerca de 50 imagens.

---

## 2. Estrutura do projeto

```
detection/
├── TRAINING_ROADMAP.md          este documento
├── requirements.txt
├── configs/
│   ├── dataset.yaml             caminhos, classes, política de imagens sem rótulo, divisão
│   ├── base_train.yaml          configuração-base de todo treino (épocas, aumento padrão, margem)
│   ├── optuna.yaml              E6: espaço de busca e configuração do estudo
│   └── experiments/
│       01_baseline.yaml  02_models.yaml  03_imgsz.yaml  04_clahe.yaml
│       05_augmentation.yaml  06b_optuna_best.yaml  07_cv.yaml  08_final.yaml
├── pavedet/                     funções reutilizáveis
│   ├── config.py                YAML, herança (extends), overrides (--set chave=valor)
│   ├── io_utils.py              imagens (HEIC/EXIF), rótulos YOLO, limpeza, dHash
│   ├── dataset.py               leitura do dataset de origem e do anotador
│   ├── splitting.py             divisão estratificada
│   ├── preprocess.py            CLAHE
│   ├── training.py              monta a configuração (herança), treina, valida e registra
│   ├── metrics.py               P/R/F1/mAP por classe e ponto de operação
│   ├── evaluation.py            avaliação completa e tempo de inferência
│   ├── tracking.py              registro (run_meta.json, registry.jsonl), configuração vigente
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
2. A **configuração vigente** dos experimentos listados em `inherit_best` (seção 3).
3. `overrides` do experimento
4. A variante que está sendo testada
5. `--set chave=valor` na linha de comando

Com isso, cada etapa parte automaticamente do vencedor das anteriores. No E2, `model: "{family}s.pt"` vira `yolo26s.pt` ou `yolo11s.pt`, conforme a família que venceu no E1.

### 2.2 Referências sem treinar de novo (`include_best`)

A configuração vigente não é treinada de novo em cada etapa. A chave `include_best` traz as execuções dela para a comparação. Exemplo: no E2, o modelo n do E1 aparece como `n (E1)`; no E3, o 960 aparece como `imgsz 960 (E1/E2)`.

### 2.3 O que é registrado

Cada execução tem a pasta `results/experiments/<experimento>/<variante>/s<seed>_f<fold>/` com os arquivos da Ultralytics (`weights/`, `results.csv`, curvas), o `run_meta.json` (configuração completa, ambiente, tempos, melhor época, métricas gerais e por classe) e `val_best/`. Cada execução também acrescenta uma linha em `results/registry.jsonl`, que só cresce.

Uma execução já concluída é pulada (dá para interromper e continuar). `--force` roda de novo, preservando a pasta antiga.

---

## 3. Como decidir: fold 0 e margem de adoção

As etapas E1 a E6 decidem com **um treino por variante, no fold 0** (cerca de 50 imagens de validação). Um único número tem ruído: na rodada anterior, a mesma configuração variou cerca de ±0,015 de mAP50-95 só trocando a seed. Por isso:

- **Margem de adoção (`adopt_margin: 0.015` no `base_train.yaml`):** uma mudança só é adotada se o mAP50-95 subir **mais de 0,015** em relação à configuração vigente. Abaixo disso é empate, e fica a vigente, que é a mais simples ou barata (testada antes).
- **A herança aplica a regra sozinha:** percorre os experimentos na ordem do pipeline e só troca a configuração vigente quando a nova supera pela margem.
- **O `compare.py` mostra o veredito** de cada variante: **melhor** (subiu mais que a margem), **pior** (caiu mais que a margem) ou **empate**.
- **A validação cruzada (E7) confere no fim** se as escolhas se sustentam em folds que não participaram de nenhuma decisão.

Para forçar outra escolha (por exemplo, preferir um modelo mais leve), passe-a com `--set` nos experimentos seguintes e registre a decisão no relatório.

---

## 4. Pipeline

| # | Etapa | Variantes | Treinos |
|---|---|---|---|
| E0 | Ambiente, auditoria, preparo, validação dos dados | — | — |
| E1 | Linha de base: família | yolo26n × yolo11n (960, padrões) | 2 |
| E2 | Tamanho | s e m da família vencedora (o n vem do E1) | 2 |
| E3 | Resolução | 1280 e 1600 no melhor modelo (o 960 vem de antes) | 2 |
| E4 | CLAHE | clipLimit 1.0 e 2.0 | 2 |
| E5 | Aumento de dados | luz forte, geometria, mixup, close_mosaic 20 | 4 (+1 combinação opcional) |
| E6 | Optuna (hiperparâmetros de treino) | otimizador, lr0, lrf, box, cls | 12 trials + 1 (E6b) |
| E7 | Validação cruzada (3 folds) | vencedora nos folds 1 e 2 (fold 0 já existe) | 2 |
| E8 | Final + teste | treino+validação, épocas da CV, 3 seeds; teste uma vez | 3 |

Cerca de **30 treinos**. Depois de cada experimento: `python scripts/compare.py --experiments <nome>`.

### E0 — Ambiente e dados

```bash
cd detection
pip install -r requirements.txt
python scripts/00_check_env.py --smoke                 # versões, device e 1 época no COCO8
bash tests/smoke_test.sh --quick                       # confere preparo e dry-runs (sem treinos)
python scripts/01_audit_dataset.py                     # auditoria da nova rotulação: results/audit/
python scripts/02_prepare_dataset.py --resplit         # divisão nova (3 folds) → splits/pavement_v3.csv
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 1.0   # → data/pavement_v3_clahe1
python scripts/02_prepare_dataset.py --preprocess clahe --clahe-clip 2.0   # → data/pavement_v3_clahe2
python scripts/03_validate_dataset.py --ultralytics
```

Confira na auditoria que as imagens sem `.txt` (`results/audit/unlabeled.jpg`) são mesmo sem dano. Depois, versione o `splits/pavement_v3.csv` no git, para todos usarem a mesma divisão.

### E1 — Linha de base: família da YOLO

| | |
|---|---|
| Variantes | `yolo26n` e `yolo11n`, `imgsz 960`, padrões da Ultralytics, `epochs 100`, `patience 25`, `batch 8` |
| Por quê | Ponto de partida e escolha da família. O 960 já se mostrou melhor que 640 na rodada anterior. |
| Decisão | Fica a de maior mAP50-95 (as duas têm o mesmo custo). |

```bash
python scripts/run_experiment.py 01_baseline
python scripts/compare.py --experiments 01_baseline
```

### E2 — Tamanho do modelo

| | |
|---|---|
| Variantes | `s` e `m` da família vencedora (o `n` entra como referência, vindo do E1) |
| Por quê | Mais capacidade pode ajudar com rachaduras e remendos. `l` e `x` ficam de fora: com cerca de 150 imagens tendem a decorar o treino, e custam 3 a 4 vezes o tempo do n. |
| Decisão | Um tamanho maior só é adotado se superar o n por mais de 0,015. |

```bash
python scripts/run_experiment.py 02_models
python scripts/compare.py --experiments 02_models
```

### E3 — Resolução

| | |
|---|---|
| Variantes | `imgsz_1280` e `imgsz_1600` no melhor modelo do E1/E2 (o 960 entra como referência) |
| Por quê | Rachaduras finas e buracos pequenos somem em resoluções menores; as fotos têm até 12 MP. Foi o único ganho consistente da rodada anterior (640 → 960). |
| Custo | O 1280 leva cerca de 1,8 vez o tempo do 960; o 1600, cerca de 2,8 vezes. Se faltar memória com o m a 1600: `--set train.batch=4`. |

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

### E5 — Aumento de dados

O aumento padrão (já no `base_train.yaml`) é o da Ultralytics: `mosaic 1.0`, `close_mosaic 10`, `scale 0.5`, `translate 0.1`, `fliplr 0.5`, `hsv_h 0.015`, `hsv_s 0.7`, `hsv_v 0.4`, sem giro, perspectiva, `flipud` ou `mixup`. Ele entra como referência. Cada variante muda uma coisa:

| Variante | Mudança | Por quê |
|---|---|---|
| `luz_forte` | `hsv_v 0.6`, `hsv_s 0.8` | Sombra de árvore, sol forte e asfalto molhado |
| `geometria` | `degrees 5`, `perspective 0.0005` | Celular inclinado e ângulos diferentes em relação ao chão |
| `mixup` | `mixup 0.1` | Regularização: menos "decoreba" com poucas imagens |
| `close_mosaic_20` | `close_mosaic 20` | Mais épocas finais com imagens normais; foi o 3º parâmetro mais importante no Optuna da rodada anterior |

Ficam de fora: `flipud` (foto em perspectiva de cabeça para baixo não acontece), `scale` mais forte e "sem aumento" (perderam na rodada anterior).

Se duas ou mais variantes ganharem ("melhor" no `compare.py`), descomente a variante `combo` no fim do `05_augmentation.yaml` com os valores delas e rode `--variants combo`.

O aumento vem **antes** do Optuna porque o melhor `lr0` e os pesos da perda dependem de quanto aumento existe.

```bash
python scripts/run_experiment.py 05_augmentation
python scripts/compare.py --experiments 05_augmentation
```

### E6 — Optuna (hiperparâmetros de treino)

| | |
|---|---|
| Base | A configuração vigente ao fim do E1–E5 |
| Espaço de busca | `optimizer` (SGD/AdamW), `lr0`, `lrf`, `box`, `cls`. Na rodada anterior, `cls` e `lr0` foram de longe os mais importantes; momentum, weight decay, warmup e cos_lr ficaram perto de zero e ficam no padrão. |
| Configuração | 12 trials, 50 épocas e paciência 15 por trial, poda pela mediana, no fold 0 |
| E6b | O melhor trial é treinado de novo com 100 épocas (`06b_optuna_best`), para comparar de igual para igual com a vigente. Só é adotado se superar por mais de 0,015. |

```bash
python scripts/tune_optuna.py
python scripts/tune_optuna.py --report        # importâncias e histórico em results/optuna/hparams_v3/
python scripts/run_experiment.py 06b_optuna_best
python scripts/compare.py --experiments 06b_optuna_best
```

### E7 — Validação cruzada (3 folds)

| | |
|---|---|
| O que faz | Treina a configuração vigente ao fim do E1–E6 nos folds 1 e 2. O `cv_report.py` junta com o fold 0, já treinado. |
| Saídas | `results/cv/cv_report.md`: mAP50-95, mAP50, P, R, F1 em **média ± desvio de 3 folds** (o número principal do relatório), por classe; a mediana da melhor época (usada no E8) e o limiar de confiança que maximiza o F1. |
| Leitura | O fold 0 participou de todas as decisões; os folds 1 e 2 não. Se a média deles ficar bem abaixo do fold 0, as escolhas se ajustaram demais ao fold 0 — registre isso no relatório. |

```bash
python scripts/run_experiment.py 07_cv
python scripts/cv_report.py
```

### E8 — Treino definitivo e teste

| | |
|---|---|
| Configuração | A vencedora, treinada em **treino + validação** (todas as imagens menos o teste), com o número de épocas fixo na mediana da melhor época da CV e sem early stopping. 3 seeds. |
| Por quê | Cerca de 50% a mais de imagens que o treino de um fold, o que pesa num dataset deste tamanho. O número de épocas vem da CV porque não sobra validação independente. Vale o modelo da última época (`last.pt`). |
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

## 5. Épocas

| Onde | Épocas (teto) | Paciência (early stopping) |
|---|---|---|
| E1 a E5, E6b e E7 | 100 | 25 |
| Trials do Optuna (E6) | 50 | 15, com poda |
| E8 (treino + validação) | mediana da melhor época da CV | 0 (sem early stopping) |

Na prática, os treinos param antes do teto: na rodada anterior, a melhor época ficou entre 20 e 70.

---

## 6. Tempo estimado (A6000, 6 vCPUs)

Tempo por treino a 960, a partir da rodada anterior: n ≈ 6 min, s ≈ 8 min, m ≈ 12 min. O 1280 leva cerca de 1,8 vez e o 1600 cerca de 2,8 vezes isso. As etapas E4 em diante usam o modelo e a resolução vencedores, então o total depende deles:

| Vencedor do E1–E3 | Total aproximado |
|---|---|
| n ou s a 960 | ~3 h |
| n ou s a 1280 | ~4 h |
| n ou s a 1600, ou m a 1280 | ~6 h |
| m a 1600 | ~8 h |

Já configurado para acelerar: `cache: disk` (a CPU não redecodifica o JPEG a cada época; o gargalo da rodada anterior era a CPU), `workers: 6`, `max_side: 1600` e o early stopping.

Se o tempo apertar:

1. Optuna com menos trials: `python scripts/tune_optuna.py --n-trials 8`
2. Aumento de dados só com 2 variantes: `--variants luz_forte close_mosaic_20`
3. Final com 1 seed: `python scripts/run_experiment.py 08_final --seeds 0` (o teste fica sem desvio)

Antes de começar, rode um treino curto para medir o tempo por época na máquina: `python scripts/train.py --name teste --set train.epochs=10`.

---

## 7. Ordem resumida

| # | Comando |
|---|---|
| E0 | `00_check_env.py --smoke` → `01_audit_dataset.py` → `02_prepare_dataset.py --resplit` → `02_prepare_dataset.py --preprocess clahe --clahe-clip 1.0` e `2.0` → `03_validate_dataset.py --ultralytics` |
| E1 | `run_experiment.py 01_baseline` → `compare.py --experiments 01_baseline` |
| E2 | `run_experiment.py 02_models` → `compare.py --experiments 02_models` |
| E3 | `run_experiment.py 03_imgsz` → `compare.py --experiments 03_imgsz` |
| E4 | `run_experiment.py 04_clahe` → `compare.py --experiments 04_clahe` |
| E5 | `run_experiment.py 05_augmentation` → `compare.py --experiments 05_augmentation` |
| E6 | `tune_optuna.py` → `tune_optuna.py --report` → `run_experiment.py 06b_optuna_best` → `compare.py --experiments 06b_optuna_best` |
| E7 | `run_experiment.py 07_cv` → `cv_report.py` |
| E8 | `run_experiment.py 08_final` → `final_test.py --experiment 08_final --confirm --conf cv` |
| Relatório | `compare.py --experiments 01_baseline 02_models 03_imgsz 04_clahe 05_augmentation 06b_optuna_best --out resumo` |

---

## 8. Métricas

| Métrica | Uso |
|---|---|
| **mAP50-95** | Métrica de seleção. Média do AP em IoU de 0,50 a 0,95: cobra caixas precisas. |
| mAP50 | AP com IoU ≥ 0,5: o modelo achou o dano mais ou menos no lugar certo |
| P, R, F1 | No limiar que maximiza o F1 (`best_conf`) |
| P_op, R_op, F1_op | No limiar fixo do ponto de operação, com matriz de confusão própria. É o que o usuário vê com `predict`. |
| Por classe | Todas as anteriores por classe. Rachadura costuma ser a mais baixa (extensão da caixa ambígua). |

Referência: em datasets públicos grandes de pavimento (milhares de imagens), modelos bem treinados ficam em torno de 0,5–0,65 de mAP50. Para cerca de 170 imagens bem anotadas, 0,35–0,45 de mAP50 no teste seria um bom resultado.

---

## 9. Comandos úteis

| Comando | Para quê |
|---|---|
| `python scripts/run_experiment.py --list` | Lista os experimentos |
| `python scripts/run_experiment.py 03_imgsz --dry-run` | Mostra a configuração herdada sem treinar |
| `python scripts/run_experiment.py 05_augmentation --variants luz_forte` | Só uma variante |
| `python scripts/run_experiment.py 03_imgsz --set train.batch=4` | Muda um valor em todas as variantes |
| `python scripts/train.py --name teste --set train.epochs=10` | Treino avulso, também registrado |
| `python scripts/evaluate.py --best-of 05_augmentation` | Avaliação detalhada na validação: matrizes, curvas, piores imagens, velocidade |
| `bash tests/smoke_test.sh` | Teste do pipeline inteiro num dataset sintético, na CPU (sem tocar em `data/` e `results/`) |

O `evaluate.py` recusa o teste. O teste só é usado pelo `final_test.py`.

---

## 10. Limitações a relatar

- **Dataset pequeno:** cerca de 170 imagens; folds com ~50 imagens de validação e teste com ~25.
- **Decisões num fold só:** cada escolha do E1 ao E6 usa o fold 0, com uma margem de 0,015 contra o ruído. A CV de 3 folds confere no fim, mas as escolhas podem ter se ajustado ao fold 0.
- **Imagens sem anotação como fundo:** se alguma delas tiver dano, o modelo aprende a ignorá-lo e a avaliação conta o acerto como falso positivo.
- **Divisão por imagem, sem agrupar a mesma cena:** há fotos do mesmo dano tiradas com segundos de diferença. Uma pode ficar no treino e outra na validação ou no teste; as métricas tendem a ficar acima do desempenho em ruas novas. O agrupamento pode ser religado em `configs/dataset.yaml` (`hash_threshold: 70`, `time_window_s: 30`, `seq_window: 2`).
- **Rachadura:** mesmo com o guia, a extensão de uma rachadura é ambígua; espera-se mAP50-95 mais baixo nessa classe.

---

## 11. Pendências antes do primeiro treino

1. Apagar os YAMLs que não fazem mais parte do pipeline: `configs/experiments/04_augmentation.yaml`, `05_imbalance.yaml`, `06_ablation.yaml`, `06b_optuna_confirm.yaml` e `07_ablation.yaml` (`git rm`).
2. Apagar ou mover a pasta `results/` antiga da máquina de treino, se existir.
3. Manter a mesma máquina e as mesmas versões em todos os experimentos.
4. Rodar `bash tests/smoke_test.sh --quick` uma vez na máquina de treino.
