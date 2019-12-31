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
##LOAD_PREFIX=meta_fixstage
LOAD_PREFIX=meta_randmask
MODE=single_task
###MODE=single_task_full_emb
###MODE=single_task_avg_emb
FINETUNE=z
EVAL_START=5
EVAL_END=101
EVAL_STEP=5
##TEST_SAMPLES=10

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_composite_v4
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

export KSHOTSAMPLE=0
export STAGE=-1

export EXP_SUFFIX=""


OLDIFS=$IFS
IFS=','

for op in  mul+replace-ith-jth+swap ; do # div+replace-ith-next+swap mul+replace-ith+reverse div+replace-ith+shift add+replace-ith-jth+swap add+replace-ith-jth+reverse mod+replace-ith-next+swap add+replace-ith-next+swap mul+replace-ith-jth+swap ; do 
   export TEST_FILE=$op


   for ts in  20 ; do
   	export TEST_SAMPLES=$ts

	if [ $MODE == "task_agnostic" ]; then
	   export LOAD=${LOAD_PREFIX}_${TEST_FILE}

	   ./scripts/eval_composition.sh
	else
	   for zsize in 128 64 32 16 ; do # 64 32 16 ; do
	   	export ZSIZE=${zsize}
#		for lr in 1e-3 2e-3 ; do
#			for zlr in 5e-2 1e-1 ; do
#				export ZLR=$zlr
#				export LOAD=${LOAD_PREFIX}_${TEST_FILE}_zsize${ZSIZE}_${ts}shot_lr${lr}_zlr${zlr}
#		
#				./scripts/eval_composition.sh
#			done
#			sleep $[ ( $RANDOM % 5 ) + 1 ]s
#		done

	   	export LOAD=${LOAD_PREFIX}_${TEST_FILE}_zsize${ZSIZE}

	   	./scripts/eval_composition.sh
#
#		sleep $[ ( $RANDOM % 6 ) + 1 ]s
	   done
	fi	
   done
done
	
IFS=$OLDIFS
	

