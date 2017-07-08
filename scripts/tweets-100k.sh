config='tweets-100k.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
export OMP_NUM_THREADS=18
export MKL_NUM_THREADS=18
python ../code/main.py $config

# Postprocesing results

