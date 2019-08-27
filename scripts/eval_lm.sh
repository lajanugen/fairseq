# Feed-forward model, multi-tasking
MODE=single_task
FINETUNE=yes

MDL=default

## DEFALUTS ##
export CKPT_DIR=/home/llajan/b6/fsl
DBG_MODE=yes
LAYERS=4
ZSIZE=128

export CUDA_VISIBLE_DEVICES=4

echo $LOAD

INIT_MDL=$LOAD/checkpoint1.pt

RUN="./scripts/run_lm.sh"

EXP_NAME=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')

mkdir -p $CKPT_DIR/$EXP_NAME

ARGS="-e $EXP_NAME -m $MODE --mdl $MDL -f -l $LAYERS"

if [ $FINETUNE == "no" ]; then
    ARGS="$ARGS --no-training"
elif [ $FINETUNE == "z" ]; then
    ARGS="$ARGS -z --lr 1e-2 --zsize $ZSIZE --task-emb-init zeros"
fi

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

if [ ! -z $INIT_MDL ]; then
    ARGS="$ARGS -i $INIT_MDL"
fi

echo $ARGS

$RUN $ARGS -t 0 --eval

rm /checkpoint/llajan/$EXP_NAME -rf
