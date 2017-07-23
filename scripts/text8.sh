config='text8.yaml'

# Preprocessing data
# python ../code/preprocess-text.py $config

# Run
export MKL_NUM_THREADS=12
export OMP_NUM_THREADS=12
python ../code/main.py $config

# Postprocesing results

