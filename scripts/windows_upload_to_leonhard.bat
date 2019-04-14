REM  REMOVE THE '--exclude=/data' the very first time you execute this bash script! --include=/data/processed/all.csv


cd C:\Users\Thomas\Desktop\deeplearning
rsync -avzh -P --stats --timeout=60 --exclude=.git --exclude=/.env  --exclude=.idea --exclude=/data --exclude=/scripts/patch_job.sh --include=/data/processed/all.csv  . grethoma@login.leonhard.ethz.ch:~/deeplearning/
cd C:\Users\Thomas\Desktop\deeplearning\scripts