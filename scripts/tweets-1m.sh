config='tweets-1m.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
export OMP_NUM_THREADS=10
export MKL_NUM_THREADS=10
python ../code/main.py $config
# python ../code/model_neg.py $config

# Postprocesing results

