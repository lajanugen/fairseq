# Feed-forward model, multi-tasking
EXP_NAME=ff_lm
MODE=single_task

MDL=default

## DEFALUTS ##
export CKPT_DIR=./checkpoints
DBG_MODE=yes
LAYERS=4
ZSIZE=128
NUMGRADS=25

export CUDA_VISIBLE_DEVICES=0

RUN="./scripts/run_lm.sh"

ARGS="-e $EXP_NAME -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 100"

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

if [[ $MODE == *"meta"* ]]; then
    ARGS="$ARGS --zlr 1e-2 --zsize $ZSIZE --numgrads $NUMGRADS"
fi

echo $EXP_NAME
echo $ARGS

$RUN $ARGS
