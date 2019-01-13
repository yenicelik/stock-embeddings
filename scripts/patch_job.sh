module load python_gpu/3.6.4
export PYTHONPATH="${PYTHONPATH}:~/"
bsub -R "rusage[mem=16000, ngpus_excl_p=1]" python ~/deeplearning/main.py
sleep 10
bjobs