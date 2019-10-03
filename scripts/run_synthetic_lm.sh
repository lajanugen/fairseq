EXP_NAME=test
TRAINING_MODE=multitask
MODEL='meta'
FINETUNE=0
TRAIN_ONLY_Z=0
LR=1e-3
ZLR=1e-3
INIT_MDL=""
CLUSTER=0
NUM_GRAD_UPDATES=100
EVAL_TASK_ID=0
TASK_EMB_SIZE=128
TASK_EMB_INIT=mean
MAX_EPOCH=10
TENSORBOARD=0
LOGLOSS=0
META_NUM_EX=4
LAYERS=1
NO_TRAINING=0
EVAL=0
DISABLE_VALID=0

SEEN_NUM_TRAIN=500
UNSEEN_NUM_TRAIN=500
UNSEEN_NUM_TEST=500

VOCAB_SIZE=100
MAX_TASKS=2000
NUM_TRAIN_TASKS=500
NUM_TEST_TASKS=64
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
    --task-emb-init)
    TASK_EMB_INIT=$2
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
    MAX_SEQ_LEN=$2
    shift; shift; ;;
    --maxtasks)
    MAX_TASKS=$2
    shift; shift; ;;
    --meta-ex)
    META_NUM_EX=$2
    shift; shift; ;;
    -l)
    LAYERS=$2
    shift; shift; ;;
    --novalid)
    DISABLE_VALID=1
    shift; ;;
esac
done


ARGS=" --training_mode $TRAINING_MODE --lr $LR --task_emb_size $TASK_EMB_SIZE --encoder_layers $LAYERS"
if [ "$FINETUNE" == "1" ]; then 
    ARGS="$ARGS --train_unseen_task --save-interval 1000 --max-epoch $MAX_EPOCH --eval_task_id $EVAL_TASK_ID --num_train $UNSEEN_NUM_TRAIN";
else
    ARGS="$ARGS --max-epoch $MAX_EPOCH --sample_num_tasks 64 --num_train $SEEN_NUM_TRAIN"; 
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
  if [ $EVAL == "0" ]; then
    RUN="python fairseq_cli/train.py"
	else
    RUN="python train_multiple_tasks.py"
	fi
else
  RUN="srun --job-name=$EXP_NAME --output=$CKPT_DIR/$EXP_NAME/train.log --error=$CKPT_DIR/$EXP_NAME/train.stderr --open-mode=append --unbuffered "
  if [ $EVAL == "0" ]; then
    RUN="$RUN python fairseq_cli/train.py"
	else
    RUN="$RUN python train_multiple_tasks.py"
    # RUN="python train_multiple_tasks.py"
	fi
  #RUN="srun --nodes=1 --gres=gpu:1 --partition=learnfair --time=30 python fairseq_cli/train.py"
fi

if [ $LOGLOSS == "1" ]; then
 	ARGS="$ARGS --log_losses $CKPT_DIR/$EXP_NAME/losses.txt"
fi

#if [ $MODEL == "snail" ]; then
#	ARGS="$ARGS --task task_suite_snail --arch cls_seq --meta_num_ex $META_NUM_EX"
#elif [ $MODEL == "matching" ]; then
#	ARGS="$ARGS --task task_suite_matching --arch cls_match --meta_num_ex $META_NUM_EX"
#elif [ $MODEL == "matching_ctx" ]; then
#	ARGS="$ARGS --task task_suite_matching --arch cls_match --contextualized --meta_num_ex $META_NUM_EX"
#else
#	ARGS="$ARGS --task task_suite_v2 --arch cls_v2"
#fi
ARGS="$ARGS --task synthetic_lm_task --arch review_tf"

ARGS="$ARGS \
	/tmp/data/ \
	--dataset-impl raw \
	--save-dir $CKPT_DIR/$EXP_NAME \
	--load_tasks_file /checkpoint/annl/transfer_learn_lm/tasks.txt \
	--max-tokens 4096 \
	--optimizer adam \
	--encoder_type transformer \
	--num_test $UNSEEN_NUM_TEST \
	--vocab_size $VOCAB_SIZE \
	--max_seq_len $MAX_SEQ_LEN \
	--num_train_tasks $NUM_TRAIN_TASKS \
	--num_test_tasks $NUM_TEST_TASKS \
	--max_tasks $MAX_TASKS \
	--task_emb_cond_type cls_token \
	--clip-norm 5 \
	--reset-dataloader \
	--z_lr $ZLR"

mkdir -p $CKPT_DIR/$EXP_NAME

echo $ARGS | tee $CKPT_DIR/$EXP_NAME/params.txt

#$RUN $ARGS | tee $CKPT_DIR/$EXP_NAME/run_log.txt

if [ $CLUSTER == "0" ]; then
	$RUN $ARGS
else

#sbatch --job-name=$EXP_NAME --nodes=1 --gres=gpu:volta:1 -C volta32gb --partition=learnfair \
sbatch --job-name=$EXP_NAME --nodes=1 --gres=gpu:volta:1 --partition=learnfair \
	--output=$CKPT_DIR/$EXP_NAME/train.log --error=$CKPT_DIR/$EXP_NAME/train.stderr --open-mode=append --time=4320 \
	--no-requeue --ntasks-per-node=1 --cpus-per-task=8 --mem=32000 \
	--wrap="$RUN $ARGS"
fi
