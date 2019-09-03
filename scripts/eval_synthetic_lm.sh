echo $LOAD

VOCAB_SIZE=12
SEQ_LEN=5

INIT_MDL=$LOAD/checkpoint1.pt

RUN="./scripts/run_synthetic_lm.sh"

#EXP_NAME=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')
EXP_NAME="$LOAD-$MODE-$MDL-$FINETUNE-$TEST_SAMPLES"

mkdir -p $CKPT_DIR/$EXP_NAME

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -f -l $LAYERS --max-epoch 100 --zsize $ZSIZE"

if [ $FINETUNE == "no" ]; then
    ARGS="$ARGS --no-training"
elif [ $FINETUNE == "z" ]; then
    ARGS="$ARGS -z --lr 1e-2 --task-emb-init zeros"
fi

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

if [ ! -z $INIT_MDL ]; then
    ARGS="$ARGS -i $INIT_MDL"
fi

echo $ARGS

$RUN $ARGS --eval --test-samples $TEST_SAMPLES -t 0

#rm /checkpoint/llajan/$EXP_NAME -rf
