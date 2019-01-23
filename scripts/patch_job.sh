module unload python_gpu/3.6.4
module load python_gpu/3.6.4
export PYTHONPATH="${PYTHONPATH}:~/"
chmod 777 ~/deeplearning/dl/main.py

bsub -R "rusage[mem=16000, ngpus_excl_p=8]" python ~/deeplearning/dl/main.py --production
#python ~/deeplearning/dl/main.py
sleep 10
bjobs