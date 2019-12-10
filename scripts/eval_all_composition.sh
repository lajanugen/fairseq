# Feed-forward model, LM training
#LOAD_PREFIX=task_agnostic
#MODE=task_agnostic
##FINETUNE=no
#FINETUNE=yes
#EVAL_START=5
#EVAL_END=101
#EVAL_STEP=5
##TEST_SAMPLES=10

# Feed-forward model, multitask training
#LOAD_PREFIX=multitask
#MODE=single_task
#FINETUNE=yes
#EVAL_START=5
#EVAL_END=101
#EVAL_STEP=5
#TEST_SAMPLES=10

# Feed-forward model, meta training
LOAD_PREFIX=meta
MODE=single_task
FINETUNE=z
EVAL_START=5
EVAL_END=101
EVAL_STEP=5
##TEST_SAMPLES=10

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_composite_v2
export DBG_MODE=no
export LAYERS=4
export ZSIZE=128

#export LOAD=$LOAD
export MODE=$MODE
export FINETUNE=$FINETUNE
export MDL=default
export EVAL_START=$EVAL_START
export EVAL_END=$EVAL_END
export EVAL_STEP=$EVAL_STEP
#export TEST_SAMPLES=$TEST_SAMPLES

#export CUDA_VISIBLE_DEVICES=0


OLDIFS=$IFS
IFS=','

for testfile in add,0 replace-ith-jth,1 reverse,2 ; do
   set -- $testfile
   export TEST_FILE=$1
   export STAGE=$2

   for ts in 1 5 10 20 ; do
   	export TEST_SAMPLES=$ts

	if [ $MODE == "task_agnostic" ]; then
	   export LOAD=${LOAD_PREFIX}_${TEST_FILE}

	   ./scripts/eval_composition.sh
	else
	   for zsize in 128 64 32 16 ; do
	   	export ZSIZE=${zsize}

	   	export LOAD=${LOAD_PREFIX}_${TEST_FILE}_zsize${ZSIZE}

	   	./scripts/eval_composition.sh

		sleep $[ ( $RANDOM % 6 ) + 1 ]s
	   done
	fi	
   done
done
	
IFS=$OLDIFS
	

