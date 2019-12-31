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
#LOAD_PREFIX=meta
LOAD_PREFIX=meta_randmask
#LOAD_PREFIX=meta_fixstage
MODE=single_task
###MODE=single_task_full_emb
###MODE=single_task_avg_emb
FINETUNE=z
EVAL_START=5
EVAL_END=100
EVAL_STEP=5
##TEST_SAMPLES=10

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_composite_grid
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

export EXP_SUFFIX=""


OLDIFS=$IFS
IFS=','

for testfile in first,0 second,1 third,2 ; do
   set -- $testfile
   export TEST_FILE=$1
   export STAGE=$2

   for ts in 1 5 10 20 ; do
   	export TEST_SAMPLES=$ts

	if [ $MODE == "task_agnostic" ]; then
	   export LOAD="sample68_${LOAD_PREFIX}"

	   ./scripts/eval_composition_grid.sh
	else
	   for zsize in 128 ; do # 64 32 16 ; do
	   	export ZSIZE=${zsize}

	   	export LOAD="sample68_${LOAD_PREFIX}_zsize${ZSIZE}"

	   	./scripts/eval_composition_grid.sh

		sleep $[ ( $RANDOM % 6 ) + 1 ]s
	   done
	fi	
   done
done
	
IFS=$OLDIFS
	

