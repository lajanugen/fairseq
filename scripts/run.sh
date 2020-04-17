# Save dir
CKPT_DIR=/home/llajan/b6/fcls6
EXP_NAME=test
TRAINING_MODE=multitask
LOAD_TASKS=/home/llajan/b6/task_split/tasks/
# LOAD_TASKS=scripts/task_split/zeroshot_v3/


# Model & hyperparameters
MODEL=default
LR=1e-3
MAX_EPOCH=10

# Task embedding parameters
ZLR=1e-2
TASK_EMB_SIZE=128
NUM_GRAD_UPDATES=10
TASK_EMB_COND_TYPE=encoder
COMPOSITIONAL=0

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
# MAX_TASKS=2000
MAX_TASKS=1200
VOCAB_SIZE=12
SEQ_LEN=5
# NUM_TRAIN_TASKS=1000
# NUM_TRAIN_TASKS=16
# NUM_TRAIN_TASKS=20
NUM_TRAIN_TASKS=40
# NUM_TRAIN_TASKS=60
NUM_TEST_TASKS=65
SEEN_NUM_TRAIN=500
UNSEEN_NUM_TRAIN=500
UNSEEN_NUM_TEST=500

CLUSTER=0
LAYERS=4

DATASET=amazon
SHOTS=5

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
    -c|--cluster)
    CLUSTER=1
    shift; ;;
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
    --test-samples)
    UNSEEN_NUM_TRAIN=$2
    shift; shift; ;;
    --train-tasks)
    NUM_TRAIN_TASKS=$2
    shift; shift; ;;
    --max-epoch)
    MAX_EPOCH=$2
    shift; shift; ;;
    --tb)
    TENSORBOARD=1
    shift; ;;
    --logloss)
    LOGLOSS=1
    shift; ;;
    --vocab)
    VOCAB_SIZE=$2
    shift; shift; ;;
    --seqlen)
    SEQ_LEN=$2
    shift; shift; ;;
    --maxtasks)
    MAX_TASKS=$2
    shift; shift; ;;
    --data)
    DATASET=$2
    shift; shift; ;;
    --shots)
    SHOTS=$2
    shift; shift; ;;
    -l)
    LAYERS=$2
    shift; shift; ;;
    --novalid)
    DISABLE_VALID=1
    shift; ;;
    --compositional)
    COMPOSITIONAL=1
    shift; ;;
		--load-tasks)
		LOAD_TASKS=$2
		shift; shift; ;;
esac
done

ARGS=" --training_mode $TRAINING_MODE --lr $LR --task_emb_size $TASK_EMB_SIZE --encoder_layers $LAYERS"

if [ $RUN_MODE == "eval" ]; then
    ARGS="$ARGS --train_unseen_task --save-interval 1000 --max-epoch $MAX_EPOCH --no_training --shots $SHOTS";
else
    ARGS="$ARGS --max-epoch $MAX_EPOCH --save-interval 20 --num_train $SEEN_NUM_TRAIN"; 
fi
if [ "$TRAIN_ONLY_Z" == "0" ]; then ARGS="$ARGS --tune_model_params"; fi

if [ "$NO_TRAINING" == "1" ]; then ARGS="$ARGS --no_training"; fi

if [ "$TENSORBOARD" == "1" ]; then ARGS="$ARGS --tensorboard-logdir $CKPT_DIR/$EXP_NAME"; fi
	
if [ "$DISABLE_VALID" == "1" ]; then ARGS="$ARGS --disable-validation"; fi

if [ ! -z $INIT_MDL ]; then
  ARGS="$ARGS --restore-file $CKPT_DIR/$INIT_MDL --reset-optimizer"
fi
ARGS="$ARGS --num_grad_updates $NUM_GRAD_UPDATES"

if [ $CLUSTER == "0" ]; then
  RUN="python fairseq_cli/train.py"
else
  RUN="srun --job-name=$EXP_NAME --output=$CKPT_DIR/$EXP_NAME/train.log --error=$CKPT_DIR/$EXP_NAME/train.stderr --open-mode=append --unbuffered "
   RUN="$RUN python fairseq_cli/train.py"
    # RUN="python train_multiple_tasks.py"
  #RUN="srun --nodes=1 --gres=gpu:1 --partition=learnfair --time=30 python fairseq_cli/train.py"
fi

if [ $LOGLOSS == "1" ]; then
 	ARGS="$ARGS --log_losses $CKPT_DIR/$EXP_NAME/losses.txt"
fi

ARGS="$ARGS --task task_textcls"
if [ $RUN_MODE == "train" ]; then
  # ARGS="$ARGS --max-sentences 150 --sample_num_tasks 5"
  # ARGS="$ARGS --max-sentences 4 --sample_num_tasks 200"
  # ARGS="$ARGS --max-sentences 20 --sample_num_tasks 50"
  # ARGS="$ARGS --max-sentences 16 --sample_num_tasks 100"
  # ARGS="$ARGS --max-sentences 16 --sample_num_tasks 50"
  # ARGS="$ARGS --max-tokens 256 --sample_num_tasks 32"
  # ARGS="$ARGS --max-sentences 200 --sample_num_tasks 50"
  ARGS="$ARGS --max-sentences 200 --sample_num_tasks 50"
  # ARGS="$ARGS --max-sentences 160 --sample_num_tasks 10"
  # ARGS="$ARGS --max-sentences 100 --sample_num_tasks 100"
  # ARGS="$ARGS --max-sentences 50 --sample_num_tasks 200"
  # ARGS="$ARGS --max-sentences 40 --sample_num_tasks 300"
  # ARGS="$ARGS --max-sentences 1000 --sample_num_tasks 10"
	ARGS="$ARGS --regularization"
else
  ARGS="$ARGS --max-sentences 16 --sample_num_tasks 40 --disable-validation"
fi
if [ $COMPOSITIONAL == "1" ]; then ARGS="$ARGS --arch cls_comp"; else ARGS="$ARGS --arch cls_v2"; fi
# ARGS="$ARGS --arch cls_v1"
# --task_emb_cond_type token/norm/adapt

#   --max-tokens 2048 \
ARGS=" \
  --save-dir $CKPT_DIR/$EXP_NAME \
  --task_descriptions_dir $CKPT_DIR/$EXP_NAME \
  --optimizer adam \
  --encoder_type transformer \
  --num_test $UNSEEN_NUM_TEST \
  --vocab_size $VOCAB_SIZE \
  --max_seq_len $SEQ_LEN \
  --num_train_tasks $NUM_TRAIN_TASKS \
  --num_test_tasks $NUM_TEST_TASKS \
  --max_tasks $MAX_TASKS \
  --batch_version \
  --clip-norm 5 \
  --normalize_loss \
  --reset-dataloader \
  --z_lr $ZLR \
  --task_emb_cond_type cls_token \
  --load_tasks $LOAD_TASKS \
	--dataset $DATASET \
	--validate-interval 10 \
  $ARGS"
# 	--disable-validation \
#   --task_emb_cond_type cls_token \

mkdir -p $CKPT_DIR/$EXP_NAME

if [ $RUN_MODE == "train" ]; then
  echo $ARGS | tee $CKPT_DIR/$EXP_NAME/params.txt
fi

$RUN $ARGS
