RUN=./scripts/run.sh

export CUDA_VISIBLE_DEVICES=0

EXP_NAME=hufp_emb300_tasks50_sent200_l1_1000ep 

$RUN -e $EXP_NAME -m meta --mdl default --zsize 300 --tb --max-epoch 1000 --zlr 1e-2 --numgrads 25 -l 1 --data huffpost
