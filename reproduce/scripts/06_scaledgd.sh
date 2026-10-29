#!/bin/bash
# Tensor RPCA by scaled gradient descent (Dong et al. 2023): Optuna on the development variables, then frozen. CPU, 8 threads.
# Our run selected r_space,r_month,r_year,z0,z1,eta,decay =
#   134.0,5.0,23.0,0.06736777226768229,0.019074469252475506,0.2775066772815105,0.6859553595833972
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A24_scaledgd.py --tune --threads 8 --trials 100 --db A24_optuna.db --out A24_tune.csv
BEST=$($PY -c "import optuna,os; s=optuna.load_study(study_name='A24_scaledgd_dev', storage='sqlite:///'+os.path.abspath('A24_optuna.db')); p=s.best_trial.params; print(','.join(repr(float(p[k])) for k in ['r_space','r_month','r_year','z0','z1','eta','decay']))")
echo "selected: $BEST"
$PY -u $SRC/A24_scaledgd.py --eval --threads 8 --source ncep --products lftx,pres,omega,pottmp,gpcp --seeds 0,1,2 --params "$BEST" --out A24_eval.csv
$PY -u $SRC/A24_scaledgd.py --eval --threads 8 --source era5 --products air,slp,pres,pottmp,rhum,omega,cape --seeds 0,1,2 --params "$BEST" --out A24_eval.csv
$PY -u $SRC/A24_scaledgd.py --eval --threads 8 --source ncep --products air,slp,rhum --seeds 0,1,2 --params "$BEST" --out A27_scaledgd_ncepdev.csv
$PY -u $SRC/A24_scaledgd.py --eval --threads 8 --source 20cr --products air,slp,pres,pottmp,rhum,omega,cape --seeds 0,1,2 --params "$BEST" --out A25_scaledgd.csv
