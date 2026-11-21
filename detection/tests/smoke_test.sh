#!/usr/bin/env bash
# Teste de fumaça do pipeline inteiro num dataset SINTÉTICO, na CPU, em poucos minutos.
# Não toca em data/, results/ nem splits/ do projeto: tudo vai para uma pasta temporária
# (variável PAVEDET_WORKDIR). Treinos de 1–2 épocas em imagens 160 px: só confere que
# os scripts funcionam de ponta a ponta, não mede desempenho.
#
# Uso (dentro da pasta detection/, com o ambiente virtual ativo):
#   bash tests/smoke_test.sh            # completo (~5–10 min na CPU)
#   bash tests/smoke_test.sh --quick    # sem treinos (só dados e dry-runs)
set -euo pipefail
cd "$(dirname "$0")/.."
TMP="${SMOKE_DIR:-$(mktemp -d -t pavedet_smoke_XXXX)}"
export PAVEDET_WORKDIR="$TMP/work"
DS="$TMP/src/dataset.yaml"
# configuração mínima para os treinos de teste
FAST="device=cpu train.epochs=2 train.imgsz=160 train.batch=8 train.workers=0 train.plots=false train.patience=0"
step() { echo; echo "=================== $*"; }

step "0. ambiente";            python scripts/00_check_env.py || true
step "1. dataset sintético";   python tests/make_synthetic_dataset.py --out "$TMP/src" --n 48
step "2. auditoria";           python scripts/01_audit_dataset.py --config "$DS" --out "$PAVEDET_WORKDIR/results/audit"
step "3. preparo";             python scripts/02_prepare_dataset.py --config "$DS"
step "3b. preparo de novo (deve reusar o manifesto)"; python scripts/02_prepare_dataset.py --config "$DS"
step "3c. preparo CLAHE (mesma divisão)"; python scripts/02_prepare_dataset.py --config "$DS" --preprocess clahe
step "4. validação";           python scripts/03_validate_dataset.py --config "$DS" --ultralytics
step "4b. validação CLAHE";    python scripts/03_validate_dataset.py --config "$DS" --name pavement_v2_clahe
step "5. dry-runs";            for e in 01 02 04 05; do python scripts/run_experiment.py $e --dry-run --set $FAST; done
python scripts/run_experiment.py --list
if [[ "${1:-}" == "--quick" ]]; then echo "OK (quick) — $TMP"; exit 0; fi

step "6. baseline (3 seeds)";  python scripts/run_experiment.py 01_baseline --set $FAST
step "6b. de novo: deve PULAR as concluídas"; python scripts/run_experiment.py 01_baseline --set $FAST
step "7. modelos (2 variantes)"; python scripts/run_experiment.py 02_models --variants yolo26n yolo11n --set $FAST
step "8. imgsz (herda o melhor do 02)"; python scripts/run_experiment.py 03_imgsz --variants imgsz_640 --set $FAST
step "9. aumento + CLAHE";     python scripts/run_experiment.py 04_augmentation --variants control rotate_5 clahe --set $FAST
step "10. desbalanceamento";   python scripts/run_experiment.py 05_imbalance --variants control rfs_t05 patch_x2 --set $FAST
step "11. optuna (3 trials) + confirmação"
cat > "$TMP/optuna_test.yaml" <<EOF
study_name: smoke
inherit_best: 05_imbalance
experiment: 06_optuna
n_trials: 3
sampler: {type: tpe, seed: 1, n_startup_trials: 2}
pruner: {type: median, n_startup_trials: 1, n_warmup_steps: 0}
trial_overrides: {train: {epochs: 2}}
search_space:
  train.optimizer: {type: categorical, choices: [SGD, AdamW]}
  train.lr0: {type: float, low: 1.0e-4, high: 1.0e-2, log: true}
  train.cls: {type: float, low: 0.3, high: 1.5}
EOF
python scripts/tune_optuna.py --config "$TMP/optuna_test.yaml" --set $FAST
python scripts/tune_optuna.py --config "$TMP/optuna_test.yaml" --report
python scripts/tune_optuna.py --config "$TMP/optuna_test.yaml" --confirm-top 2 --seeds 0 --set $FAST
step "12. ablação (2 variantes)"; python scripts/run_experiment.py 07_ablation --variants full minus_model --seeds 0 --set $FAST
step "13. final (2 seeds)";    python scripts/run_experiment.py 08_final --seeds 0 1 --set $FAST train.epochs=2
step "14. comparação";         python scripts/compare.py
step "15. avaliação (val)";    python scripts/evaluate.py --best-of 08_final --bench 5
step "16. teste final";        python scripts/final_test.py --experiment 08_final --confirm
step "16b. teste de novo sem --rerun: deve RECUSAR"
if python scripts/final_test.py --experiment 08_final --confirm; then echo "FALHOU: o teste deveria estar bloqueado"; exit 1; fi
step "17. predict";            python scripts/predict.py --run "$(ls -d $PAVEDET_WORKDIR/results/experiments/08_final/final/s0_f0)" --source "$TMP/src/dataset/detection/images/IMG_1005.jpg"
echo; echo "SMOKE TEST OK — saídas em $TMP"
