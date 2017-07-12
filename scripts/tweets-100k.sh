config='tweets-100k.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
export OMP_NUM_THREADS=12
export MKL_NUM_THREADS=12
python ../code/main.py $config
# python ../code/main_neg.py $config

# Postprocesing results

