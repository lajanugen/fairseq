echo $LOAD

VOCAB_SIZE=12
SEQ_LEN=5

INIT_MDL=$LOAD/checkpoint1.pt

RUN="./scripts/run_composition.sh"

EXP_NAME="$LOAD-$MODE-$MDL-$FINETUNE-$TEST_SAMPLES-$TEST_FILE$EXP_SUFFIX"

export TEST_FILE="tasks-$TEST_FILE.test.150.txt"

mkdir -p $CKPT_DIR/$EXP_NAME

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -f -l $LAYERS --max-epoch 100 --zsize $ZSIZE --stage $STAGE --eval-iter $EVAL_START $EVAL_END $EVAL_STEP "

if [ $FINETUNE == "no" ]; then
    ARGS="$ARGS --no-training"
elif [ $FINETUNE == "z" ]; then
    ARGS="$ARGS -z --lr 1e-2 "
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
