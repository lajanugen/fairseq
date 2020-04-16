RUN=./scripts/run.sh

export CUDA_VISIBLE_DEVICES=0

EXP_NAME=hufp_emb300_tasks50_sent200_l1_1000ep
# 1-shot - Train: 1.00, Test: 0.51, Oracle: 0.53
# 5-shot - Train: 0.80, Test: 0.41, Oracle: 0.44

$RUN -e $EXP_NAME -m meta --mdl default --zsize 300 --max-epoch 100 --zlr 1e-2 --numgrads 100 -l 1 --runmode eval --data huffpost --shots 1
