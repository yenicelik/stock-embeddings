rsync -r -v --exclude ~/deeplearning/venv/ ~/deeplearning/ yedavid@login.leonhard.ethz.ch:~/deeplearning/ --progress

#rsync -v --include=dl/ --include=scripts --include=requirements.txt ~/deeplearning/ yedavid@login.leonhard.ethz.ch:~/deeplearning/ --progress

# TODO: NOTICE: REMOVE THE `--exclude ~/deeplearning/data/` the very first time you execute this bash script!

rsync -rnv \
--include=/data/model_kaggle_basepath.pkl \
yedavid@login.leonhard.ethz.ch:~/deeplearning/data/ ~/deeplearning/data/ --progress
