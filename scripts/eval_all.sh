# Feed-forward model, LM training
LOAD=ff_taskagnostic
MODE=task_agnostic
FINETUNE=no
MDL=default
TEST_SAMPLES=10

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_lm
export DBG_MODE=no
export LAYERS=4
export ZSIZE=128

export LOAD=$LOAD
export MODE=$MODE
export FINETUNE=$FINETUNE
export MDL=$MDL
export TEST_SAMPLES=$TEST_SAMPLES

./scripts/eval_synthetic_lm.sh
