module load python_gpu/3.6.4
export PYTHONPATH="${PYTHONPATH}:~/"
bsub -W 32:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python ~/deeplearning/dl/main.py
bsub -I -R "rusage[mem=16000, ngpus_excl_p=1]" python ~/deeplearning/dl/main.py
sleep 10
bjobs