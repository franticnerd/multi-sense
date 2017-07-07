config='tweets-1m.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
OMP_NUM_THREADS=10
python ../code/main.py $config

# Postprocesing results

