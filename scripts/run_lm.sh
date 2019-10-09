# Save dir
CKPT_DIR=/home/llajan/b6/fsl
EXP_NAME=test
TRAINING_MODE=multitask

# Model & hyperparameters
MODEL=default
LAYERS=4
LR=1e-3
MAX_EPOCH=10

# Task embedding parameters
ZLR=1e-2
TASK_EMB_SIZE=8
NUM_GRAD_UPDATES=10

# Logging
DISABLE_VALID=0
TENSORBOARD=0
LOGLOSS=0

# Eval
INIT_MDL=""
EVAL=0
FINETUNE=0
TRAIN_ONLY_Z=0
NO_TRAINING=0
EVAL_TASK_ID=0

# Task settings
MAX_TASKS=15000
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
    --eval)
    EVAL=1
    shift; ;;
    -f|--fntn)
    FINETUNE=1
    shift; ;;
	  --no-training)
    NO_TRAINING=1
    shift; ;;
    -z|--zonly)
    TRAIN_ONLY_Z=1
    shift; ;;
    --lr) 
	  LR=$2
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

if [ $EVAL == "0" ]; then
  RUN="python fairseq_cli/train.py"
else
  RUN="python train_multiple_tasks_v2.py"
  EXP_NAME=$(head /dev/urandom | tr -dc A-Za-z0-9 | head -c 13 ; echo '')
fi

ARGS=" --training_mode $TRAINING_MODE --lr $LR --task_emb_size $TASK_EMB_SIZE --encoder_layers $LAYERS"

if [ "$FINETUNE" == "1" ]; then 
    ARGS="$ARGS --train_unseen_task --save-interval 1000 --max-epoch $MAX_EPOCH --eval_task_id $EVAL_TASK_ID";
else
    ARGS="$ARGS --max-epoch $MAX_EPOCH";
fi
if [ "$TRAIN_ONLY_Z" == "0" ]; then ARGS="$ARGS --tune_model_params"; fi

if [ "$NO_TRAINING" == "1" ]; then ARGS="$ARGS --no_training"; fi

if [ "$TENSORBOARD" == "1" ]; then ARGS="$ARGS --tensorboard-logdir $CKPT_DIR/$EXP_NAME"; fi

if [ "$DISABLE_VALID" == "1" ]; then ARGS="$ARGS --disable-validation"; fi

if [ "$FASTEVAL" == "yes" ]; then ARGS="$ARGS --fast-eval --eval-num-iter 10"; fi
	
if [ ! -z $INIT_MDL ]; then
  ARGS="$ARGS --restore-file $CKPT_DIR/$INIT_MDL --reset-optimizer"
fi

if [ $LOGLOSS == "1" ]; then ARGS="$ARGS --log_losses $CKPT_DIR/$EXP_NAME/losses.txt"; fi

ARGS="$ARGS --task review_task --arch review_tf"

ARGS="$ARGS \
  /home/llajan/b6/amazon_reviews_v2/ \
  --dataset-impl raw \
  --save-dir $CKPT_DIR/$EXP_NAME \
  --max-tokens 4096 \
  --optimizer adam \
  --encoder_type transformer \
  --max_seq_len $MAX_SEQ_LEN \
  --max_tasks $MAX_TASKS \
  --task_emb_cond_type cls_token \
  --clip-norm 5 \
  --reset-dataloader \
  --num_grad_updates $NUM_GRAD_UPDATES \
  --z_lr $ZLR"

mkdir -p $CKPT_DIR/$EXP_NAME

echo $ARGS | tee $CKPT_DIR/$EXP_NAME/params.txt

#$RUN $ARGS | tee $CKPT_DIR/$EXP_NAME/run_log.txt
$RUN $ARGS

if [ $EVAL == "1" ]; then
    if [ ! -z $EXP_NAME ]; then
        rm $CKPT_DIR/$EXP_NAME -rf
    fi
fi
