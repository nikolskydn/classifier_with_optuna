#!/bin/sh
set -e
PYTHONPATH=`pwd` python  training/run_training.py
# python test.py --model catboost
# python test.py --model xgboost
# python test.py --model lightgbm
# python test.py --model stacking
# python rfecvs.py
exec "$@"
