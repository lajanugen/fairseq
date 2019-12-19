VOCAB_SIZE=12
SEQ_LEN=5

RUN="./scripts/run.sh"

ARGS="-e $EXP_NAME --vocab $VOCAB_SIZE --seqlen $SEQ_LEN -m $MODE --mdl $MDL -l $LAYERS --tb --max-epoch 100 --zsize $ZSIZE"

if [[ $MDL == "snail" || $MDL == "matching" ]]; then
	  ARGS="$ARGS --meta-ex $((TEST_SAMPLES+1))"
fi

if [[ $MODE == *"meta"* || $MODE == *"maml"*  ]]; then
#    ARGS="$ARGS --zlr 1e-2 --numgrads $NUMGRADS"
    ARGS="$ARGS --zlr $ZLR --lr $LR --numgrads $NUMGRADS"
fi

if [[ $MODE == "maml_meta" ]]; then
    ARGS="$ARGS --task-emb-init maml "
fi

if [ $DBG_MODE == "no" ]; then
    ARGS="$ARGS -c"
fi

echo $EXP_NAME
echo $ARGS

$RUN $ARGS
