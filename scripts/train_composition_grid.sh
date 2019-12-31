RUN="./scripts/run_composition_grid.sh"

VOCAB_SIZE=200
SEQ_LEN=25

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 100 --maxtasks 2000 --zsize $ZSIZE "

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

if [[ $MODE == *"meta"* ]]; then
    ARGS="$ARGS --zlr 1e-2 --numgrads $NUMGRADS "
fi

if [[ $LOG_LOSS == 1 ]]; then
    ARGS="$ARGS --logloss "
fi

echo $EXP_NAME
echo $ARGS

$RUN $ARGS
