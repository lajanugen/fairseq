# Save dir
CKPT_DIR=/home/llajan/b6/fsl
EXP_NAME=test
TRAINING_MODE=multitask

# Model & hyperparameters
MODEL=default
LR=1e-3
MAX_EPOCH=10

# Task embedding parameters
ZLR=1e-2
TASK_EMB_SIZE=8
NUM_GRAD_UPDATES=10
TASK_EMB_COND_TYPE=encoder

# Logging
DISABLE_VALID=0
TENSORBOARD=0
LOGLOSS=0

# Eval
INIT_MDL=""
RUN_MODE=train
TRAIN_ONLY_Z=0
NO_TRAINING=0
EVAL_TASK_ID=0
FASTEVAL=no

# Task settings
MAX_TASKS=1000
MAX_SEQ_LEN=66

while [[ $# -gt 0 ]]
do
key="$1"

case $key in
    --mdl)
    MODEL=$2
    shift; shift; ;;
    -m|--mode)
    TRAINING_MODE=$2
    shift; shift; ;;
    --runmode)
    RUN_MODE=$2
    shift; shift; ;;
    --no-training)
    NO_TRAINING=1
    shift; ;;
    -z|--zonly)
    TRAIN_ONLY_Z=1
    shift; ;;
    --lr) 
    LR=$2
    shift; shift; ;;
    --ztype)
    TASK_EMB_COND_TYPE=$2
    shift; shift; ;;
    --zlr)
    ZLR=$2
    shift; shift; ;;
    -i|--init)
    INIT_MDL=$2
    shift; shift; ;;
    -e|--exp)
    EXP_NAME=$2
    shift; shift; ;;
    --numgrads)
    NUM_GRAD_UPDATES=$2
    shift; shift; ;;
    -t|--taskid)
    EVAL_TASK_ID=$2
    shift; shift; ;;
    --zsize)
    TASK_EMB_SIZE=$2
    shift; shift; ;;
    --max-epoch)
    MAX_EPOCH=$2
    shift; shift; ;;
    --tb)
    TENSORBOARD=1
    shift; ;;
    --novalid)
    DISABLE_VALID=1
    shift; ;;
esac
done

ARGS="--task language_modeling_meta --arch transformer_lm_meta_gpt"

if [ $RUN_MODE == "eval" ]; then
    # RUN="python train_multiple_tasks_v2.py"
    RUN="python train_multiple_tasks_v3.py"
    EXP_NAME=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')
elif [ $RUN_MODE == "sample" ]; then
    RUN="python fairseq_cli/train.py"
    # ARGS="$ARGS --LMinit"
else
    RUN="python fairseq_cli/train.py"
    ARGS="$ARGS --LMinit"
fi

ARGS="$ARGS --training_mode $TRAINING_MODE --lr $LR --encoder_embed_dim $TASK_EMB_SIZE --mdl $MODEL"

if [ "$RUN_MODE" == "train" ]; then 
    ARGS="$ARGS --sample-break-mode complete_doc --max-epoch $MAX_EPOCH";
elif [ "$RUN_MODE" == "sample" ]; then 
    ARGS="$ARGS --train_unseen_task --sample-break-mode eos --save-interval $MAX_EPOCH --max-epoch $MAX_EPOCH --eval_task_id $EVAL_TASK_ID --dropout 0.0 --attention-dropout 0.0";
else
    ARGS="$ARGS --train_unseen_task --sample-break-mode eos --save-interval 1000 --max-epoch $MAX_EPOCH --eval_task_id $EVAL_TASK_ID --dropout 0.0 --attention-dropout 0.0";
fi

if [ "$TRAIN_ONLY_Z" == "1" ]; then ARGS="$ARGS --train_only_z"; fi

if [ "$NO_TRAINING" == "1" ]; then ARGS="$ARGS --no_training"; fi

if [ "$TENSORBOARD" == "1" ]; then ARGS="$ARGS --tensorboard-logdir $CKPT_DIR/$EXP_NAME"; fi

if [ "$DISABLE_VALID" == "1" ]; then ARGS="$ARGS --disable-validation"; fi

if [ "$FASTEVAL" == "yes" ]; then ARGS="$ARGS --fast-eval --eval-num-iter 1"; fi
	
if [ ! -z $INIT_MDL ]; then
    ARGS="$ARGS --restore-file $CKPT_DIR/$INIT_MDL --reset-optimizer"
fi

if [ $LOGLOSS == "2" ]; then ARGS="$ARGS --log_losses $CKPT_DIR/$EXP_NAME/losses.txt"; fi

#	--criterion adaptive_loss \
#	--ddp-backend=no_c10d \
#  --dataset-impl lazy \
#  /home/llajan/fairseq/scripts/task_v3_bpe/ \
ARGS="$ARGS \
    /home/llajan/b6/amazon_reviews_v3/ \
    --criterion cross_entropy \
    --dataset-impl raw \
    --save-dir $CKPT_DIR/$EXP_NAME \
    --max-tokens 1024 \
    --optimizer adam \
    --max_tasks $MAX_TASKS \
    --clip-norm 5 \
    --reset-dataloader \
    --num_grad_updates $NUM_GRAD_UPDATES \
    --task_emb_cond_type $TASK_EMB_COND_TYPE \
    --save-interval-updates 10000 \
    --z_lr $ZLR"

mkdir -p $CKPT_DIR/$EXP_NAME

echo $ARGS | tee $CKPT_DIR/$EXP_NAME/params.txt

$RUN $ARGS

if [ $RUN_MODE == "eval" ]; then
    if [ ! -z $EXP_NAME ]; then
        rm $CKPT_DIR/$EXP_NAME -rf
    fi
fi
