config='tweets-100k.yaml'

# Preprocessing data
# python ../code/preprocess.py $config

# Run
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
python ../code/main.py $config

# Postprocesing results

