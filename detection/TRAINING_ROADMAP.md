# Roteiro de treinamento — Detecção de danos em pavimentos com YOLO

Este roteiro cobre o pipeline completo para treinar um detector de três classes de dano em pavimento: rachadura (crack), remendo (patch) e buraco (pothole). Os dados de origem estão em `dataset/detection` e todo o código fica na pasta `detection/` do repositório.

Nenhum treino foi executado com o dataset real. Este documento descreve o que deve ser rodado, em que ordem, por quê, e como decidir cada passo.

---

## 1. Análise inicial

### 1.1 Documentos de referência

O manual do aluno (*Treinando a YOLO para detecção de objetos*) e a aula 10 (*Pré-processamento e aumento de dados*) foram usados como base. A tabela mostra o que foi adotado, o que foi ajustado e por quê.

| Recomendação | O que foi feito | Justificativa |
|---|---|---|
| Começar com `yolo26n.pt`, `imgsz=640`, `epochs=100`, `patience=20` | Adotado como linha de base, com `epochs=150` e `patience=30` | Com cerca de 100 imagens de treino há poucas iterações por época. O early stopping encerra antes se não houver melhora. |
| mAP50-95 como métrica principal | Adotado para escolher entre variantes. P, R e F1 também são registrados, por classe e em dois limiares de confiança | É o mesmo critério que a Ultralytics usa para salvar o `best.pt`, o que mantém a seleção coerente. |
| Dividir por grupo para evitar vazamento | Implementado (`pavedet/splitting.py`), mas **desligado por decisão do grupo** (P3, seção 1.4) | A divisão é feita por imagem. O agrupamento pode ser religado em `configs/dataset.yaml`. |
| Usar o teste uma única vez | Implementado com uma trava (`final_test.py`) | Impede que o teste vire validação sem que ninguém perceba. |
| Uma mudança por vez, contra uma linha de base, com seed fixa | Adotado em todos os experimentos | Acrescentado: o ruído entre seeds é medido e usado como critério para avançar (seção 3). |
| `optimizer=auto` ignora o `lr0` | O Optuna fixa o otimizador junto com o `lr0` | Sem isso, a busca de `lr0` não teria efeito nenhum. |
| Escolher o aumento de dados pelo domínio | `flipud` excluído; giro pequeno, luz, escala e perspectiva testados | As fotos são em perspectiva (rua e céu). Uma foto espelhada na vertical não acontece no uso real. |
| CLAHE e outros pré-processamentos devem ser testados | O CLAHE virou uma variante do experimento E4, aplicada no preparo e reaplicada no `predict.py` | O pré-processamento precisa ser idêntico no treino, na validação, no teste e no uso. |
| Matriz de confusão do `val()` | Não é usada para P/R/F1 no ponto de operação. Uma matriz própria é calculada com `conf=0.25` | O `val()` gera a matriz com `conf=0.001`, o limiar usado no cálculo do mAP, que não representa o uso real. |

### 1.2 Ambiente

| Item | Situação encontrada |
|---|---|
| Repositório | Usar só o clone `computer-vision` (atualizar com `git pull`). A antiga `computer-vision-main` era um ZIP desatualizado e foi apagada. |
| Scripts existentes | `scripts/train.py`, `bench.py` e `split.py` são de **classificação**: usam `top1` e `split-folders` sem grupos. Não foram alterados. |
| Onde treinar | Defina antes de começar: no Mac (GPU via `mps`), no Colab (CUDA) ou na máquina de quem tiver GPU NVIDIA. O pipeline escolhe sozinho entre `cuda`, `mps` e `cpu`. |
| Bibliotecas | Veja `detection/requirements.txt`. A versão `ultralytics>=8.4` é necessária para a YOLO26. |

### 1.3 Auditoria do dataset

A auditoria foi gerada por `scripts/01_audit_dataset.py`. O relatório completo fica em `detection/results/audit/report.md`.

| Item | Valor |
|---|---|
| Imagens | 173: 141 com `.txt` e 32 sem |
| Caixas | 645 (média de 4,6 por imagem anotada) |
| Ordem das classes (`labels.txt`) | `0` rachadura · `1` remendo · `2` buraco |
| Instâncias por classe | rachadura 366 (56,7%) · buraco 200 (31,0%) · remendo 79 (12,2%) |
| Imagens que contêm a classe | rachadura 118 · buraco 84 · remendo 49 |
| Formatos | 93 `.jpeg`, 74 `.jpg` e 6 `.heic` |
| Resolução | de 0,17 a 16 MP (mediana de 1,2 MP), 10 resoluções diferentes, 7 fotos com orientação EXIF girada |
| Menor lado da caixa em `imgsz=640` (mediana) | rachadura 80 px · remendo 61 px · buraco 44 px |

| Anotador | Imagens | Com `.txt` | Rachadura | Remendo | Buraco |
|---|---|---|---|---|---|
| andressa | 35 | 25 | 64 | 16 | 37 |
| claudio | 31 | 23 | 58 | 12 | 45 |
| duda | 36 | 30 | 58 | 20 | 36 |
| edmar | 36 | 36 | 140 | 10 | 57 |
| pamela | 35 | 27 | 46 | 21 | 25 |

### 1.4 Problemas encontrados e decisões

| # | Problema | Evidência | Decisão |
|---|---|---|---|
| P1 | **32 imagens sem anotação que não são fundo.** Várias mostram buracos e rachaduras visíveis. | `results/audit/unlabeled.jpg` | **Decisão do grupo:** usar as imagens como estão. As 32 entram no dataset como imagens sem objetos (`unlabeled_policy: background`). Consequência na seção 9. |
| P2 | **O critério de anotação muda de pessoa para pessoa.** Na classe remendo aparecem tampa de bueiro e meio-fio pintado. Na classe rachadura aparecem faixa pintada e desgaste da superfície. Algumas caixas de rachadura cobrem a largura inteira da rua, enquanto outras cobrem só um trecho. | `results/audit/crops/*.jpg`, uma linha por anotador, com o nome de cada imagem | O grupo decidiu manter as anotações como estão. Fica registrado como limitação do dataset (seção 9). |
| P3 | **A mesma cena aparece em fotos de anotadores diferentes.** As fotos foram tiradas em saídas conjuntas, com o mesmo celular, e divididas depois. O mesmo buraco aparece em fotos de duas pessoas com segundos de diferença. | Horários do EXIF em `results/audit/images.csv` | **Decisão do grupo:** dividir por imagem, sem agrupar fotos da mesma cena (agrupamento desligado em `configs/dataset.yaml`). Consequência na seção 9. |
| P4 | Classe remendo minoritária (12%) | Contagens acima | Experimento E5 (oversampling apenas no treino). |
| P5 | 6 imagens HEIC, das quais 1 tem nome diferente do seu `.txt` (`C35BCB13-…_1_201_a.heic` × `C35BCB13-…963.txt`) | Overlay conferido em `results/audit/overlays/` | Convertidas para JPEG no preparo. O nome é corrigido por `label_aliases` no config, e as caixas conferem com a imagem. |
| P6 | 2 caixas degeneradas (largura 0) em `9EEC03FD-…` | Relatório da auditoria | Removidas na limpeza. O original não é alterado. |
| P7 | Fotos com orientação EXIF girada | 7 imagens | O preparo aplica a orientação na imagem antes de salvar. Os overlays confirmam que os rótulos do Make Sense seguem a imagem já girada. |
| P8 | Validação e teste pequenos (cerca de 25 imagens cada) | 173 imagens | As métricas variam muito entre execuções. Por isso: 3 seeds no baseline para medir o ruído, um critério de avanço baseado nesse ruído e folds opcionais para comparações apertadas. |
| P9 | O dHash não detecta a mesma cena vista de outro ângulo | O vizinho mais próximo fica sempre a 87 bits ou mais | O dHash é usado só para detectar cópias idênticas. O agrupamento depende da hora e da numeração (P3). |

---

## 2. Estrutura do projeto

```
detection/
├── TRAINING_ROADMAP.md          este documento
├── requirements.txt             bibliotecas
├── .gitignore                   data/, results/ e *.pt não vão para o git
├── configs/
│   ├── dataset.yaml             caminhos, classes, aliases, política de imagens sem rótulo, divisão
│   ├── base_train.yaml          configuração-base de todo treino (aumento de dados explícito)
│   ├── optuna.yaml              espaço de busca e configuração do estudo
│   └── experiments/
│       ├── 01_baseline.yaml     02_models.yaml     03_imgsz.yaml
│       ├── 04_augmentation.yaml 05_imbalance.yaml  06b_optuna_confirm.yaml
│       └── 07_ablation.yaml     08_final.yaml
├── pavedet/                     funções reutilizáveis (o código não se repete entre scripts)
│   ├── config.py                YAML, herança (extends), overrides (--set chave=valor)
│   ├── io_utils.py              imagens (HEIC/EXIF), rótulos YOLO, limpeza, dHash
│   ├── dataset.py               leitura do dataset de origem e do anotador
│   ├── splitting.py             grupos de mesma cena e divisão estratificada por grupo
│   ├── preprocess.py            CLAHE (pré-processamento determinístico)
│   ├── training.py              monta a configuração, treina, valida e registra; oversampling
│   ├── metrics.py               P/R/F1/mAP por classe e ponto de operação com matriz própria
│   ├── evaluation.py            avaliação completa e tempo de inferência
│   ├── tracking.py              registro (run_meta.json, registry.jsonl), preservação, melhor variante
│   ├── plots.py                 gráficos
│   └── env.py                   device, versões, git, memória
├── scripts/
│   ├── 00_check_env.py          01_audit_dataset.py   02_prepare_dataset.py   03_validate_dataset.py
│   ├── run_experiment.py        train.py (treino avulso)   tune_optuna.py
│   ├── compare.py               evaluate.py   final_test.py   predict.py
├── tests/
│   ├── make_synthetic_dataset.py    dataset falso para testar os scripts
│   └── smoke_test.sh                teste de ponta a ponta (CPU, em pasta temporária)
├── splits/pavement_v2.csv       manifesto da divisão (VERSIONADO: a mesma divisão para todo o grupo)
│   (splits/pavement_v1.csv é a divisão anterior, com agrupamento e sem as imagens sem rótulo; não é mais usada)
├── data/                        dataset preparado (gerado; não vai para o git)
└── results/                     auditoria, execuções, comparações, Optuna e teste final (não vai para o git)
```

### 2.1 Como a configuração é montada

Cada execução junta as camadas abaixo, nesta ordem. A camada de baixo vence a de cima.

1. `base_train.yaml`
2. Melhor variante do experimento anterior (`inherit_best`), escolhida pela média da métrica de seleção entre as seeds. Se `inherit_best` tiver uma lista, vale o melhor entre todos os experimentos listados. O E7 e o E8 usam `[05_imbalance, 06b_optuna_confirm]`: se o Optuna não superar o E5, a configuração do E5 é herdada automaticamente.
3. `overrides` do experimento
4. A variante que está sendo testada
5. `--set chave=valor` na linha de comando

Com isso, cada experimento parte automaticamente do vencedor do anterior, sem copiar valores à mão.

### 2.2 O que é registrado em cada execução

A pasta de cada execução é `results/experiments/<experimento>/<variante>/s<seed>_f<fold>/` e contém:

- os arquivos da Ultralytics: `weights/best.pt` e `last.pt`, `results.csv`, curvas e `args.yaml`;
- `run_meta.json`, com a configuração completa, o ambiente (versões, GPU, commit do git), o tempo de treino, o pico de memória, as épocas rodadas e a melhor época, parâmetros, GFLOPs, métricas gerais e por classe e o caminho dos pesos;
- `val_best/`, com a validação do `best.pt`.

Cada execução também acrescenta uma linha a `results/registry.jsonl`. Esse arquivo só cresce, nunca é reescrito.

Para preservar resultados:

- uma execução já concluída é pulada;
- `--force` roda de novo, mas a pasta antiga é renomeada (`__anterior_<data>`), não apagada;
- uma execução interrompida é renomeada para `__incompleta_<data>`.

---

## 3. Critério geral para avançar

Com uma validação de cerca de 25 imagens, duas execuções idênticas que mudam só a seed dão resultados diferentes. O experimento E1 mede essa variação.

- **Ruído (σ):** o desvio-padrão do mAP50-95 entre as 3 seeds do baseline. O `compare.py` calcula esse valor e o mostra nos gráficos como uma faixa.
- **Uma mudança é adotada quando** a variante supera a referência por **mais de 1σ** e não piora o F1 de nenhuma classe além de 1σ.
- **Se a diferença for menor que σ,** as duas opções são consideradas empatadas. Fica a mais simples ou mais barata (modelo menor, imgsz menor, sem oversampling). Para desempatar, rode as duas finalistas com `--seeds 0 1 2` ou `--folds 0 1 2`.
- **A decisão vale também quando a resposta é "não muda nada".** Esse resultado entra no relatório, como recomenda a aula.

O `compare.py` mostra para cada variante a coluna **"supera ruído"**, com os valores `sim`, `não` ou `pior`. A referência de cada experimento é definida no próprio YAML pela chave `reference` (por exemplo, `yolo26n` no E2 e `imgsz_640` no E3). Quando o YAML não define, a referência é a variante `control`.

---

## 4. Pipeline experimental

A ordem vai das decisões que mais mudam o resultado e são mais baratas de testar para os ajustes finos e caros. Os dados vêm primeiro, porque nenhum hiperparâmetro compensa rótulos ruins ou vazamento. A arquitetura e a resolução vêm antes do aumento de dados, porque mudam a escala do problema. O Optuna fica para depois que as decisões de dados estiverem fechadas.

### Etapa 0 — Ambiente

- **Objetivo:** garantir que o treino roda e saber onde roda.
- **Script:** `scripts/00_check_env.py`
- **Critério para avançar:** nenhuma linha marcada como `FALTA`, e `device` aparece como `0` (CUDA) ou `mps`. Treinar na CPU é possível, mas muito lento.

```bash
cd "<caminho>/computer-vision/detection"
python3 -m venv ~/yolo-env
source ~/yolo-env/bin/activate
pip install -r requirements.txt
python scripts/00_check_env.py --smoke
```

O que cada linha faz:

- `cd ...`: entra na pasta do projeto. Troque `<caminho>` pelo caminho da sua máquina.
- `python3 -m venv ~/yolo-env`: cria o ambiente virtual. Só é necessário uma vez (manual, seção 1).
- `source ~/yolo-env/bin/activate`: ativa o ambiente. Repita sempre que abrir um terminal novo.
- `pip install -r requirements.txt`: instala as bibliotecas da lista.
- `python scripts/00_check_env.py --smoke`: mostra as versões e o device, e treina 1 época no COCO8 (o teste rápido do manual).

### Etapa 1 — Auditoria do dataset

- **Objetivo:** conhecer o dataset e achar problemas antes de treinar.
- **Script:** `scripts/01_audit_dataset.py`. Só lê os dados e não altera nada no dataset.
- **Resultado:** `results/audit/` (relatório, CSVs, gráficos, recortes por classe e anotador, overlays, quase-duplicatas e imagens sem rótulo).
- **Critério para avançar:** o relatório não mostra problemas novos além dos da seção 1.4.

```bash
python scripts/01_audit_dataset.py
```

Esse comando gera o relatório em `results/audit/report.md`.

### Etapa 2 — Preparação e divisão

- **Objetivo:** gerar uma cópia limpa no formato YOLO e a divisão em treino, validação e teste.
- **Script:** `scripts/02_prepare_dataset.py`
- **O que o script faz:**
  - aplica a orientação EXIF;
  - converte HEIC para JPEG;
  - reduz as imagens para no máximo 2048 px (os rótulos normalizados não mudam);
  - remove caixas inválidas;
  - separa um **teste fixo de 15%** e divide o resto em **5 folds**, por imagem (agrupamento desligado, seção 1.4) e estratificando por classe e por anotador. As imagens sem `.txt` entram como imagens sem objetos. A validação padrão é o fold 0, com cerca de 17% das imagens.
- **Arquivos gerados:** `data/pavement_v2/` (imagens, rótulos, listas, `data_fold0.yaml` a `data_fold4.yaml`, `data_trainval.yaml` e `split_report.md`) e `splits/pavement_v2.csv`.
- **Por que o manifesto vai para o git:** quem rodar o preparo depois recebe exatamente a mesma divisão. Se o dataset mudar (reanotação, imagens novas), o script recusa reaproveitar o manifesto. Nesse caso, crie uma versão nova com `--resplit --name pavement_v2`, para não misturar resultados de divisões diferentes.

```bash
python scripts/02_prepare_dataset.py
python scripts/02_prepare_dataset.py --preprocess clahe
```

O que cada linha faz:

- A primeira linha cria `data/pavement_v2` e o manifesto da divisão.
- A segunda cria `data/pavement_v2_clahe`, com CLAHE aplicado e a mesma divisão. Ela só é necessária para a variante `clahe` do E4.

**Critério para avançar:** em `split_report.md`, cada partição tem as 3 classes e a linha "Grupos que atravessam partições" mostra 0.

### Etapa 3 — Validação do dataset preparado

- **Objetivo:** encontrar erros de dados antes de gastar GPU.
- **Script:** `scripts/03_validate_dataset.py`
- **O que confere:** arquivos existentes, formato dos rótulos, classes presentes em treino e validação, nenhuma imagem ou grupo em duas partições, nenhuma cópia idêntica (md5) entre partições e leitura pela Ultralytics.
- **Critério para avançar:** o script termina com `0 erros`.

```bash
python scripts/03_validate_dataset.py --ultralytics
python scripts/03_validate_dataset.py --name pavement_v2_clahe
```

A primeira linha confere `pavement_v2` e testa se a Ultralytics consegue ler o dataset. A segunda confere a versão com CLAHE.

### Etapa 4 — E1: linha de base

| | |
|---|---|
| Objetivo | Ponto de partida e medida do ruído (σ) |
| Configuração | YOLO26n, `imgsz=640`, padrões da Ultralytics, seeds 0, 1 e 2 |
| Arquivo | `configs/experiments/01_baseline.yaml` |
| Resultado esperado | Um mAP50-95 de referência e o σ entre seeds. Remendo deve ser a classe com pior AP e F1. |
| Critério para avançar | As 3 execuções terminam. Se σ for muito alto, considere usar folds nas comparações apertadas. |

```bash
python scripts/run_experiment.py 01_baseline
python scripts/compare.py --experiments 01_baseline
```

A primeira linha treina as 3 seeds. A segunda gera a tabela e os gráficos em `results/comparisons/01_baseline/`.

### Etapa 5 — E2: versões e tamanhos da YOLO

| | |
|---|---|
| Objetivo | Escolher a família (YOLOv8, YOLO11 ou YOLO26) e o tamanho (n ou s) |
| Variantes | `yolov8n`, `yolov8s`, `yolo11n`, `yolo11s`, `yolo26n`, `yolo26s`. Opcionais: `yolo11m` e `yolo26m` |
| Justificativa | Com cerca de 100 imagens, modelos grandes tendem a decorar o treino. Os tamanhos m só valem a pena se s ganhar de n por mais de σ, o que indicaria que capacidade ajuda. |
| Resultado esperado | Diferenças pequenas entre famílias. O tamanho s deve ganhar um pouco de n, com custo maior. |
| Critério para avançar | Fica a variante de maior mAP50-95. Se a diferença for menor que σ, fica a mais leve (veja o `tradeoff_inference.png`). |

```bash
python scripts/run_experiment.py 02_models --variants yolov8n yolov8s yolo11n yolo11s yolo26n yolo26s
python scripts/compare.py --experiments 02_models
```

A primeira linha roda as 6 variantes principais, sem as opcionais m. A segunda compara as variantes, incluindo tempo de treino e de inferência.

### Etapa 6 — E3: resolução de entrada (imgsz)

| | |
|---|---|
| Objetivo | Verificar se uma resolução maior ajuda a achar rachaduras finas e buracos pequenos |
| Variantes | 640, 960 e 1280. O batch diminui junto: 16, 8 e 4. |
| Herança | Usa o melhor modelo do E2 |
| Justificativa | As fotos vão até 16 MP, e as rachaduras têm poucos pixels de largura. O custo cresce com o quadrado do imgsz. |
| Resultado esperado | Ganho de 640 para 960, especialmente em rachadura. De 960 para 1280 o ganho tende a ser menor. |
| Critério para avançar | Fica o menor imgsz que estiver a menos de σ do melhor. Confira se a memória e o tempo cabem no hardware. |

```bash
python scripts/run_experiment.py 03_imgsz
python scripts/compare.py --experiments 03_imgsz
```

### Etapa 7 — E4: aumento de dados e pré-processamento

| | |
|---|---|
| Objetivo | Medir cada técnica isoladamente contra o controle |
| Variantes | `control`, `no_aug` (mostra quanto o aumento padrão já ajuda), `rotate_5`, `light_strong`, `scale_strong`, `perspective`, `mixup_01`, `close_mosaic_20`, `clahe` |
| Justificativa | Cada técnica corresponde a uma variação que acontece no uso: celular inclinado, sombra e sol, distância, ângulo. O `flipud` ficou de fora porque não é realista. O aumento de dados acontece só no treino e é aplicado online pela Ultralytics. |
| Resultado esperado | `no_aug` deve piorar, confirmando que o aumento padrão ajuda. Os ganhos das outras variantes tendem a ser pequenos. |
| Critério para avançar | Fica a melhor variante. As técnicas que superam o ruído são anotadas, porque orientam a faixa do espaço de busca do Optuna. |

```bash
python scripts/run_experiment.py 04_augmentation
python scripts/compare.py --experiments 04_augmentation
```

Se o CLAHE for adotado, ele é reaplicado automaticamente no `predict.py`, porque fica registrado no `dataset_info.json`.

### Etapa 8 — E5: desbalanceamento

| | |
|---|---|
| Objetivo | Melhorar a classe remendo sem piorar a média |
| Variantes | `control`, `rfs_t05` e `rfs_t10` (Repeat Factor Sampling), `patch_x2` e `patch_x3` (repetição fixa das imagens com remendo), `cls_gain_1` (peso maior na perda de classe) |
| Justificativa | O oversampling age só na lista de treino. A validação e o teste continuam com as imagens originais. A classe remendo está em cerca de 35% das imagens: com limiar 0,5 do RFS as imagens com remendo entram cerca de 1,2 vez por época; com 1,0, cerca de 1,7 vez. Um limiar abaixo de 0,35 não repetiria nada. |
| Resultado esperado | O AP e o F1 de remendo sobem. Com repetição demais, o modelo pode decorar as mesmas imagens. |
| Critério para avançar | `F1__remendo` e `mAP50_95__remendo` sobem por mais de σ e a média não cai por mais de σ. Confira as colunas por classe no `compare.py`. |

```bash
python scripts/run_experiment.py 05_imbalance
python scripts/compare.py --experiments 05_imbalance
```

### Etapa 9 — E6: otimização de hiperparâmetros com Optuna

| | |
|---|---|
| Objetivo | Ajustar otimizador, taxa de aprendizado, pesos da perda e intensidade do aumento |
| Base | A melhor configuração do E5 (modelo, imgsz, aumento e desbalanceamento já decididos) |
| Configuração | `configs/optuna.yaml`: 40 trials TPE, poda pela mediana, trials de 80 épocas, objetivo na validação (fold 0) |
| Espaço de busca | `optimizer`, `lr0`, `lrf`, `momentum`, `weight_decay`, `warmup_epochs`, `cos_lr`, `box`, `cls`, `scale`, `mosaic`, `hsv_v`, `degrees`, `close_mosaic` |
| Justificativa | O TPE explora melhor que uma busca em grade. A poda interrompe cedo os trials ruins e economiza tempo. O estudo fica salvo em SQLite e pode ser retomado depois de uma interrupção. Trials curtos são ruidosos, por isso as 3 melhores configurações são treinadas de novo por completo, com 2 seeds (E6b). |
| Resultado esperado | Ganho pequeno sobre o E5. O relatório mostra a importância de cada hiperparâmetro. |
| Critério para avançar | O E7 e o E8 herdam sozinhos a configuração de maior média entre o E5 e o E6b. Se o E6b ganhar por menos de σ, é empate: o grupo pode preferir os padrões, mais simples, e para isso basta tirar o `06b_optuna_confirm` da lista `inherit_best` do `07_ablation.yaml` e do `08_final.yaml`. Relate o resultado do Optuna em qualquer caso. |

```bash
python scripts/tune_optuna.py
python scripts/tune_optuna.py --report
python scripts/tune_optuna.py --confirm-top 3 --seeds 0 1
python scripts/compare.py --experiments 05_imbalance 06b_optuna_confirm
```

O que cada linha faz:

1. Roda ou continua o estudo. `--n-trials 10` roda mais 10 trials.
2. Gera o histórico, a importância dos hiperparâmetros e `best_params.yaml` em `results/optuna/hparams_v1/`.
3. Treina por completo as 3 melhores configurações, com seeds 0 e 1, como experimento `06b_optuna_confirm`.
4. Compara as confirmadas com o vencedor do E5.

### Etapa 10 — E7: ablação

| | |
|---|---|
| Objetivo | Medir quanto cada decisão contribui, removendo uma de cada vez da configuração final |
| Variantes | `full`, `minus_optuna_hparams`, `minus_augmentation`, `minus_oversampling`, `minus_preprocessing`, `minus_imgsz`, `minus_model` |
| Configuração | 3 seeds, porque as diferenças aqui são pequenas |
| Observação | Rode só as variantes de componentes que foram adotados. Se o oversampling perdeu no E5, `minus_oversampling` fica igual a `full`. |
| Resultado esperado | Uma tabela que mostra qual decisão mais pesa. Em geral são o imgsz e o modelo. |
| Critério para avançar | A tabela entra no relatório. Se algum componente não fizer falta (diferença menor que σ), remova-o para simplificar e rode o E8 sem ele. |

```bash
python scripts/run_experiment.py 07_ablation --variants full minus_imgsz minus_model minus_augmentation
python scripts/compare.py --experiments 07_ablation
```

Ajuste a lista de `--variants` para os componentes que foram adotados.

### Etapa 11 — E8: treino definitivo

| | |
|---|---|
| Objetivo | Treinar o modelo final com a configuração vencedora |
| Configuração | `epochs=300`, `patience=50`, 3 seeds |
| Justificativa | Mais épocas e o early stopping decidindo onde parar. As 3 seeds permitem relatar média ± desvio, em vez de um único número que pode ter sido sorte. |
| Opcional | A variante `final_trainval` treina em treino e validação juntos, com o número de épocas fixo (mediana das `best_epoch`). Usa um pouco mais de dados, mas fica sem validação independente. Só use se o grupo concordar. |
| Critério para avançar | As 3 execuções terminam, com métricas de validação próximas às do E6b ou E7 `full`. |

```bash
python scripts/run_experiment.py 08_final
python scripts/evaluate.py --best-of 08_final
```

A primeira linha treina as 3 seeds. A segunda faz a avaliação detalhada na validação: matrizes de confusão, curvas, métricas por classe, piores imagens e tempo de inferência.

### Etapa 12 — Avaliação final no teste (uma vez)

- **Objetivo:** obter a nota honesta do modelo.
- **Script:** `scripts/final_test.py`
- **O que faz:** avalia **todas** as seeds do E8 no teste e relata média ± desvio. O modelo a entregar é o de melhor resultado na **validação**. Escolher pelo teste seria vazamento.
- **Trava:** a primeira execução grava `results/final/TEST_LOCK.json`. Rodar de novo exige `--rerun`, e isso fica registrado no relatório.
- **Saída:** `results/final/08_final_<data>/test_report.md`, `test_runs.csv`, a pasta de cada seed com as matrizes de confusão (contagens e normalizada), as curvas PR, F1, P e R, as métricas por classe, as piores imagens e o tempo de inferência.

```bash
python scripts/final_test.py --experiment 08_final --confirm
```

O `--confirm` é obrigatório. Ele confirma que todas as decisões já foram tomadas.

### Uso do modelo em imagens novas

```bash
python scripts/predict.py --run results/experiments/08_final/final/s0_f0 --source pasta_de_fotos --conf 0.25
```

Esse comando aplica o mesmo preparo usado no treino (EXIF, HEIC e CLAHE, se foi adotado) e salva as imagens com as caixas em `results/predictions/`. Troque `s0_f0` pela seed indicada no `test_report.md`.

---

## 5. Métricas e saídas da avaliação

| Métrica | Onde aparece | Observação |
|---|---|---|
| Precision, Recall, F1 (geral e por classe) | `run_meta.json`, `per_class.csv`, `compare.py` | Calculadas no limiar de confiança que maximiza o F1 médio (`best_conf`), o mesmo que a Ultralytics imprime |
| `P_op`, `R_op`, `F1_op` (por classe e micro) | Mesmos arquivos | Calculadas num **limiar fixo** (`op_conf: 0.25`, configurável), a partir de uma matriz de confusão própria. Mostram o que o usuário vê no `predict`. |
| mAP50 e mAP50-95 | Todos | Não dependem de limiar. O mAP50-95 é a métrica de seleção. |
| Matriz de confusão | `confusion_matrix_conf0.25*.png` | Com a linha e a coluna "fundo": objeto não encontrado (FN) e objeto inventado (FP) |
| Curvas PR, F1, P e R | `ultralytics/` dentro de cada avaliação | Geradas pela Ultralytics |
| Gráficos comparativos | `results/comparisons/<nome>/` | Barras com desvio e faixa de ruído, F1 por classe, métrica × inferência e métrica × tempo de treino |
| Análise de erros | `per_image.csv`, `worst_images/` | Caixas reais em verde e previstas em vermelho |
| Custo computacional | `run_meta.json`, `metrics.json` | Tempo de treino, épocas, pico de memória da GPU e da RAM, parâmetros, GFLOPs, tamanho do `.pt`, ms por imagem (média, p50, p95) e FPS |

Os tempos dependem do hardware. Compare só execuções feitas na mesma máquina. O `device` fica registrado em cada execução.

---

## 6. Ordem de execução resumida

| # | Etapa | Comando principal | Execuções |
|---|---|---|---|
| 0 | Ambiente | `00_check_env.py --smoke` | — |
| 1 | Auditoria | `01_audit_dataset.py` | — |
| 2 | Preparo | `02_prepare_dataset.py` (+ `--preprocess clahe`) | — |
| 3 | Validação dos dados | `03_validate_dataset.py --ultralytics` | — |
| 4 | E1 baseline | `run_experiment.py 01_baseline` | 3 |
| 5 | E2 modelos | `run_experiment.py 02_models` | 6 (+2) |
| 6 | E3 imgsz | `run_experiment.py 03_imgsz` | 3 |
| 7 | E4 aumento | `run_experiment.py 04_augmentation` | 9 |
| 8 | E5 desbalanceamento | `run_experiment.py 05_imbalance` | 6 |
| 9 | E6 Optuna + E6b | `tune_optuna.py` e depois `--confirm-top 3` | 40 trials (curtos, com poda) + 6 |
| 10 | E7 ablação | `run_experiment.py 07_ablation` | até 21 |
| 11 | E8 final | `run_experiment.py 08_final` | 3 |
| 12 | Teste | `final_test.py --confirm` | uma vez |

Depois de cada experimento, rode `python scripts/compare.py --experiments <nome>` e aplique o critério da seção 3.

### Versão enxuta, se o tempo de GPU for curto

- E2 só com `yolo11n`, `yolo26n` e `yolo26s`.
- E3 só com 640 e 960.
- E4 só com `control`, `no_aug`, `light_strong` e `scale_strong`.
- E5 só com `control`, `rfs_t05` e `patch_x2`.
- Optuna com `--n-trials 15`.
- E7 só com `full`, `minus_imgsz` e `minus_model`.

Para isso, use `--variants` e `--seeds`.

---

## 7. Comandos úteis

| Comando | O que faz |
|---|---|
| `python scripts/run_experiment.py --list` | Lista os experimentos e quantas variantes cada um tem |
| `python scripts/run_experiment.py 02_models --dry-run` | Mostra a configuração final de cada variante, sem treinar |
| `python scripts/run_experiment.py 03_imgsz --seeds 0 1 2` | Repete com mais seeds, para desempatar |
| `python scripts/run_experiment.py 02_models --folds 0 1 2` | Valida em outros folds (validação cruzada) |
| `python scripts/run_experiment.py 02_models --set device=mps train.batch=8` | Muda a configuração na hora |
| `python scripts/train.py --name teste --set train.epochs=10` | Treino avulso, também registrado |
| `python scripts/evaluate.py --run <pasta da execução>` | Avalia uma execução específica na validação |

O `evaluate.py` recusa o teste. O teste só é usado pelo `final_test.py`.

---

## 8. O que foi testado

| Teste | Situação |
|---|---|
| Auditoria no dataset real | Rodada (só leitura). Resultados na seção 1.3. |
| Preparo, validação e treinos com o dataset real | **Não rodados** |
| Teste de ponta a ponta (`tests/smoke_test.sh`) com dataset sintético, na nuvem e na CPU, com 2 épocas | Passaram: ambiente, auditoria, preparo (com reuso do manifesto e CLAHE), validação (incluindo a leitura pela Ultralytics), dry-runs, E1 (3 seeds e pulo das concluídas), E2, E3 com herança, E4 com CLAHE, E5 com oversampling e o primeiro trial do Optuna |
| Resto do teste sintético: poda do Optuna, `--confirm-top`, E7, E8, `compare.py`, `evaluate.py`, `final_test.py`, `predict.py` | **Não chegaram a rodar**: o teste foi interrompido |

Depois do teste, todos os arquivos passaram por uma revisão de código completa (leitura, sem executar). A revisão corrigiu a herança depois do Optuna, a referência das comparações, uma variante de oversampling que não tinha efeito, o dry-run do Optuna e o formato das listas de imagens: as entradas `./../images/...` dependiam de um detalhe de versão da Ultralytics e passaram a ser `./images/...`, com as listas na raiz do dataset preparado.

Para rodar o teste sintético completo na sua máquina antes do primeiro treino real:

```bash
bash tests/smoke_test.sh
```

Esse comando roda o pipeline inteiro com dados falsos, numa pasta temporária. Não toca em `data/`, `results/` nem `splits/`. Use `--quick` para pular os treinos.

---

## 9. Limitações conhecidas do dataset

O grupo decidiu usar o dataset como está: sem reanotar, com as imagens sem anotação incluídas e com a divisão feita por imagem. Isso traz três limitações, que devem constar no relatório.

- **Imagens sem anotação usadas como fundo.** As 32 imagens sem `.txt` entram no treino, na validação e no teste como imagens sem nenhum dano. Várias delas mostram buracos e rachaduras. No treino, isso ensina o modelo a não marcar esses danos. Na avaliação, um dano encontrado numa dessas imagens conta como falso positivo.
- **Divisão por imagem, sem agrupar a mesma cena.** Há fotos do mesmo dano tiradas com segundos de diferença, muitas vezes anotadas por pessoas diferentes. Com a divisão por imagem, fotos da mesma cena podem ficar uma no treino e outra na validação ou no teste. Nesse caso, a validação e o teste avaliam o modelo em cenas que ele já viu no treino, e as métricas tendem a ficar acima do desempenho em fotos de ruas novas.
- **Critério de anotação diferente entre anotadores.** Há tampas de bueiro e meio-fio pintado anotados como remendo, e faixa pintada e desgaste da superfície anotados como rachadura. Algumas rachaduras têm uma caixa para a rua inteira, outras uma caixa por trecho. O modelo aprende exemplos contraditórios, e a avaliação compara as previsões com rótulos inconsistentes. O esperado é que remendo e rachadura tenham métricas mais baixas e mais instáveis entre seeds. A pasta `results/audit/crops/` mostra exemplos por classe e por anotador.

Para voltar atrás em qualquer uma das duas primeiras decisões, basta mudar `unlabeled_policy` ou os valores de `grouping` em `configs/dataset.yaml` e preparar uma versão nova do dataset com outro `prepare.name`.

## 10. Pendências antes do primeiro treino

1. Combinar com o grupo onde os treinos vão rodar e manter a mesma máquina e as mesmas versões em todos os experimentos.
2. Quem for treinar: rodar `bash tests/smoke_test.sh` uma vez antes dos experimentos reais (seção 8).
