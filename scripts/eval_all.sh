# Feed-forward model, LM training
LOAD=ff_taskagnostic
MODE=task_agnostic
FINETUNE=no
MDL=default

## DEFALUTS ##
export CKPT_DIR=/home/llajan/b6/fsl
export DBG_MODE=yes
export LAYERS=4
export ZSIZE=128

export LOAD=$LOAD
export MODE=$MODE
export FINETUNE=$FINETUNE
export MDL=$MDL

./scripts/eval_lm.sh
