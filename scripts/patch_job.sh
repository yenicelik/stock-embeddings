module unload python_gpu/3.6.4
module load python_gpu/3.6.4 hdf5/1.10.1
#module load python_gpu/3.6.1 hdf5/1.10.1  =>https://scicomp.ethz.ch/wiki/Leonhard_beta_testing#Hierarchical_modules

export PYTHONPATH="${PYTHONPATH}:~/"
export PYTHONPATH=$PYTHONPATH:~/deeplearning/

bsub -W 32:00 -R "rusage[mem=16000, ngpus_excl_p=1]" python ~/deeplearning/dl/main.py
bsub -I -R "rusage[mem=16000, ngpus_excl_p=1]" python ~/deeplearning/dl/main.py

bsub -I -n 20 -R "rusage[mem=4500, ngpus_excl_p=1]" python ~/deeplearning/dl/training/training_nns.py

chmod -R 777 ~/deeplearning/dl/training/training_nns.py

#From https://scicomp.ethz.ch/wiki/Getting_started_with_GPUs#Tensorflow_example
#For example, to run a serial job with one GPU,
#bsub -R "rusage[ngpus_excl_p=1]" ./my_cuda_program
#or on a full node with all eight GPUs and up to 90 GB of RAM,
#bsub -n 20 -R "rusage[mem=4500,ngpus_excl_p=8]" ./my_cuda_program
#or on two full nodes:
#bsub -n 40 -R "rusage[mem=4500,ngpus_excl_p=8] span[ptile=20]" ./my_cuda_program

#bsub -n 20 -R "rusage[mem=4500, ngpus_excl_p=8]" python ~/deeplearning/dl/main.py
#bsub -I -n 20 -R "rusage[mem=4500, ngpus_excl_p=8]" python ~/deeplearning/dl/training/training_nns.py

sleep 10
bjobs