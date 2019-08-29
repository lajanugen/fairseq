RUN="./scripts/run_lm.sh"

ARGS="-e $EXP_NAME -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 10 --maxtasks 100"

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

if [[ $MODE == *"meta"* ]]; then
    ARGS="$ARGS --zlr 1e-2 --zsize $ZSIZE --numgrads $NUMGRADS --novalid"
fi

echo $EXP_NAME
echo $ARGS

$RUN $ARGS
