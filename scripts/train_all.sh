# Feed-forward model, LM training
EXP_NAME=ff_taskagnostic
MODE=task_agnostic
MDL=default

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_lm
export DBG_MODE=no
export LAYERS=4
export ZSIZE=128
#export NUMGRADS=10
export NUMGRADS=25

export EXP_NAME=$EXP_NAME
export MODE=$MODE
export MDL=$MDL
#export CUDA_VISIBLE_DEVICES=0

./scripts/train_synthetic_lm.sh
