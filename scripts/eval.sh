VOCAB_SIZE=12
SEQ_LEN=5

echo $LOAD

INIT_MDL=$LOAD/checkpoint100.pt

RUN="./scripts/run.sh"

#RANDOM_SUFFIX=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')
EXP_NAME="$LOAD-$MODE-$MDL-$FINETUNE-$TEST_SAMPLES"

mkdir -p /checkpoint/annl/transfer_learn/$EXP_NAME

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -f -l $LAYERS --zsize $ZSIZE"

if [ $FINETUNE == "no" ]; then
    ARGS="$ARGS --no-training"
elif [ $FINETUNE == "z" ]; then
    ARGS="$ARGS -z "
    if [ $MODE == "maml_single_task" ]; then
	ARGS="$ARGS --lr 1e-1 --task-emb-init none "
    else
        ARGS="$ARGS --lr 1e-2 --task-emb-init zeros"
    fi
fi
    
if [ $MODE == "maml" ]; then
    ARGS="$ARGS --zlr $ZLR --numgrads $NUMGRADS --task-emb-init none "
fi

if [[ $MDL == "snail" || $MDL == "matching" ]]; then
	ARGS="$ARGS --meta-ex $((TEST_SAMPLES+1))"
fi

if [ ! -z $INIT_MDL ]; then
    ARGS="$ARGS -i $INIT_MDL"
fi

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

$RUN $ARGS --test-samples $TEST_SAMPLES -t 0 --eval

#cat /checkpoint/annl/transfer_learn/$EXP_NAME/train.log
#rm /checkpoint/annl/transfer_learn/$EXP_NAME -rf
