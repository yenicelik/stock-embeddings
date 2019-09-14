#rsync -v --include=dl/ --include=scripts --include=requirements.txt ~/deeplearning/ yedavid@login.leonhard.ethz.ch:~/deeplearning/ --progress

# TODO: NOTICE: REMOVE THE `--exclude ~/deeplearning/data/` the very first time you execute this bash script!

rsync -rv \
--exclude=/venv/ \
--exclude=/.git/ \
--exclude=/.idea/ \
--exclude=/data/ \
--exclude=/.env \
--include=/data/processed/all.csv \
--include=/data/processed/ \
~/deeplearning/ yedavid@login.leonhard.ethz.ch:~/deeplearning/ --progress --ignore-times


rsync -rv \
--exclude=/venv/ \
--exclude=/.git/ \
--exclude=/.idea/ \
--exclude=/data/ \
--exclude=/.env \
--include=/data/processed/all.csv \
--include=/data/processed/ \
-e 'ssh -p 2222' \
~/deeplearning/ david@77.59.149.134:~/deeplearning/ --progress --ignore-times
