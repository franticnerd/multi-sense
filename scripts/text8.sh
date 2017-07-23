config='text8.yaml'

# Preprocessing data
# python ../code/preprocess-text.py $config

# Run
export OMP_NUM_THREADS=15
export MKL_NUM_THREADS=15
python ../code/main.py $config

# Postprocesing results

