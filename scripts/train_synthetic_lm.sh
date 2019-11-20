RUN="./scripts/run_synthetic_lm.sh"

VOCAB_SIZE=12
SEQ_LEN=5

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 200 --maxtasks 2000 --zsize $ZSIZE "

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

if [[ $MODE == *"meta"* ]]; then
    ARGS="$ARGS --zlr 1e-2 --numgrads $NUMGRADS "
fi

echo $EXP_NAME
echo $ARGS

$RUN $ARGS
