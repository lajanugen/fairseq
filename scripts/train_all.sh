# Feed-forward model, LM training
EXP_NAME=ff_taskagnostic
MODE=task_agnostic
MDL=default

## DEFALUTS ##
export CKPT_DIR=/home/llajan/b6/fsl
export DBG_MODE=yes
export LAYERS=4
export ZSIZE=128
export NUMGRADS=10

export EXP_NAME=$EXP_NAME
export MODE=$MODE
export MDL=$MDL

./scripts/train_lm.sh
