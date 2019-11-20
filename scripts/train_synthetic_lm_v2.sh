RUN="./scripts/run_synthetic_lm_maml_v2.sh"

VOCAB_SIZE=12
SEQ_LEN=5

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 100 --maxtasks 2000 --zsize $ZSIZE --numgrads $NUMGRADS "

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi


echo $EXP_NAME
echo $ARGS

$RUN $ARGS
