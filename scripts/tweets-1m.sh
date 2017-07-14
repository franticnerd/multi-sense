config='tweets-1m.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
python ../code/main.py $config

# Postprocesing results

