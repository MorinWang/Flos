#!/bin/bash
# LR-STSS (Mondal et al. 2025, official SNN__LOGN_GTV solver): Optuna on the development variables, then frozen.
# Run on a CUDA GPU in the paper; our run selected psi,lda_l,lda_t =
#   0.9944356402301897,2.6664185432938476e-06,1.7182002795304627e-08
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A23_mondal.py --tune --device $FLOS_DEVICE --trials 36 --db A23_optuna.db --out A23_tune.csv
BEST=$($PY -c "import optuna,os; s=optuna.load_study(study_name='A23_mondal_dev', storage='sqlite:///'+os.path.abspath('A23_optuna.db')); p=s.best_trial.params; print('%r,%r,%r' % (p['psi'], p['lda_l'], p['lda_t']))")
echo "selected: $BEST"
$PY -u $SRC/A23_mondal.py --eval --device $FLOS_DEVICE --source ncep --products lftx,pres,omega,pottmp,gpcp --seeds 0,1,2 --params "$BEST" --out A23_eval.csv
$PY -u $SRC/A23_mondal.py --eval --device $FLOS_DEVICE --source era5 --products air,slp,pres,pottmp,rhum,omega,cape --seeds 0,1,2 --params "$BEST" --out A23_eval.csv
$PY -u $SRC/A23_mondal.py --eval --device $FLOS_DEVICE --source ncep --products air,slp,rhum --seeds 0,1,2 --params "$BEST" --out A27_mondal_ncepdev.csv
$PY -u $SRC/A23_mondal.py --eval --device $FLOS_DEVICE --source 20cr --products air,slp,pres,pottmp,rhum,omega,cape --seeds 0,1,2 --params "$BEST" --out A25_mondal.csv
