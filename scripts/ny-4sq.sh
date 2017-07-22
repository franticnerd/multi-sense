config='ny-4sq.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
export MKL_NUM_THREADS=15
export OMP_NUM_THREADS=15
python ../code/main.py $config
# python ../code/main_neg.py $config

# Postprocesing results

