RUN="./scripts/run_synthetic_lm_maml.sh"

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 200 --maxtasks 2000 --zsize $ZSIZE --lr $LR --zlr $ZLR --train-tasks $NUM_TRAIN_TASKS --numgrads $NUMGRADS "

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi


echo $EXP_NAME
echo $ARGS

$RUN $ARGS
