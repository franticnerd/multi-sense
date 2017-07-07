config='tweets-10k.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
# OMP_NUM_THREADS=10
python ../code/main.py $config

# Postprocesing results

