config='tweets-10k.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
export MKL_NUM_THREADS=2
export OMP_NUM_THREADS=2
# python ../code/main.py $config
python ../code/main_neg.py $config

# Postprocesing results

