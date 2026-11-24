#!/usr/bin/env bash
# Teste de fumaça do pipeline inteiro num dataset SINTÉTICO, na CPU.
# Não toca em data/, results/ nem splits/ do projeto: tudo vai para uma pasta temporária
# (variável PAVEDET_WORKDIR). Treinos de 2 épocas em imagens 160 px: só confere que
# os scripts funcionam de ponta a ponta, não mede desempenho.
#
# Uso (dentro da pasta detection/, com o ambiente virtual ativo):
#   bash tests/smoke_test.sh            # completo (~15-25 min na CPU)
#   bash tests/smoke_test.sh --quick    # sem treinos (só dados e dry-runs)
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="${SMOKE_DIR:-$(mktemp -d -t pavedet_smoke_XXXX)}"
export PAVEDET_WORKDIR="$TMP/work"
DS="$TMP/src/dataset.yaml"
# configuração mínima para os treinos de teste
FAST="device=cpu train.epochs=2 train.imgsz=160 train.batch=8 train.workers=0 train.plots=false train.patience=0 train.cache=false"
# dois folds bastam para testar a comparação pareada
F="--folds 0 1"
step() { echo; echo "=================== $*"; }

step "0. ambiente";            python scripts/00_check_env.py || true
step "1. dataset sintético";   python tests/make_synthetic_dataset.py --out "$TMP/src" --n 48
step "2. auditoria";           python scripts/01_audit_dataset.py --config "$DS" --out "$PAVEDET_WORKDIR/results/audit"
step "3. preparo";             python scripts/02_prepare_dataset.py --config "$DS"
step "3b. preparo de novo (deve reusar o manifesto)"; python scripts/02_prepare_dataset.py --config "$DS"
step "3c. preparo CLAHE 1.0 e 2.0 (mesma divisão)"
python scripts/02_prepare_dataset.py --config "$DS" --preprocess clahe --clahe-clip 1.0
python scripts/02_prepare_dataset.py --config "$DS" --preprocess clahe --clahe-clip 2.0
test -d "$PAVEDET_WORKDIR/data/pavement_v3_clahe1" && test -d "$PAVEDET_WORKDIR/data/pavement_v3_clahe2"
step "4. validação";           python scripts/03_validate_dataset.py --config "$DS" --ultralytics
step "4b. validação CLAHE";    python scripts/03_validate_dataset.py --config "$DS" --name pavement_v3_clahe1
step "5. dry-runs";            for e in 01 02; do python scripts/run_experiment.py $e --dry-run --set $FAST; done
python scripts/run_experiment.py --list
if [[ "${1:-}" == "--quick" ]]; then echo "OK (quick) — $TMP"; exit 0; fi

step "E1. baseline";           python scripts/run_experiment.py 01_baseline $F --set $FAST
step "E1b. de novo: deve PULAR as concluídas"; python scripts/run_experiment.py 01_baseline $F --set $FAST
step "E2. modelos";            python scripts/run_experiment.py 02_models $F --set $FAST
python scripts/compare.py --experiments 02_models
step "E3. imgsz (herda o melhor do E1/E2)"; python scripts/run_experiment.py 03_imgsz $F --set $FAST train.imgsz=192
step "E4. CLAHE";              python scripts/run_experiment.py 04_clahe $F --set $FAST
python scripts/compare.py --experiments 04_clahe
step "E5. optuna (3 trials)"
cat > "$TMP/optuna_test.yaml" <<EOF
study_name: smoke
inherit_best: [01_baseline, 02_models, 03_imgsz, 04_clahe]
experiment: 05_optuna
n_trials: 3
sampler: {type: tpe, seed: 1, n_startup_trials: 2}
pruner: {type: median, n_startup_trials: 1, n_warmup_steps: 0}
trial_overrides: {train: {epochs: 2}}
search_space:
  train.optimizer: {type: categorical, choices: [SGD, AdamW]}
  train.lr0: {type: float, low: 1.0e-4, high: 1.0e-2, log: true}
  train.mixup: {type: float, low: 0.0, high: 0.2}
  oversample.patch_factor: {type: categorical, choices: [1, 2, 3]}
EOF
python scripts/tune_optuna.py --config "$TMP/optuna_test.yaml" --dry-run --set $FAST
python scripts/tune_optuna.py --config "$TMP/optuna_test.yaml" --set $FAST
python scripts/tune_optuna.py --config "$TMP/optuna_test.yaml" --report
step "E6. ablação (variantes iguais à full são puladas)"
python scripts/run_experiment.py 06_ablation $F --set $FAST
python scripts/compare.py --experiments 06_ablation
step "E7. validação cruzada (folds 3 e 4) + relatório"
python scripts/run_experiment.py 07_cv --set $FAST
# no teste rápido os folds 0-2 do vencedor podem ter só 0 e 1: o relatório junta o que houver
python scripts/cv_report.py
step "E8. final (treino+validação, épocas da CV)"
python scripts/run_experiment.py 08_final --seeds 0 1 --set device=cpu train.imgsz=160 train.batch=8 train.workers=0 train.plots=false train.cache=false
step "resumo";                 python scripts/compare.py --experiments 01_baseline 02_models 03_imgsz 04_clahe 06_ablation --out resumo
step "avaliação (val)";        python scripts/evaluate.py --best-of 06_ablation --bench 5
step "teste final";            python scripts/final_test.py --experiment 08_final --confirm --conf cv
step "teste de novo sem --rerun: deve RECUSAR"
if python scripts/final_test.py --experiment 08_final --confirm; then echo "FALHOU: o teste deveria estar bloqueado"; exit 1; fi
step "predict";                python scripts/predict.py --run "$(ls -d $PAVEDET_WORKDIR/results/experiments/08_final/final_trainval/s0_ftrainval)" --source "$TMP/src/dataset/detection/images/IMG_1005.jpg"
echo; echo "SMOKE TEST OK — saídas em $TMP"
