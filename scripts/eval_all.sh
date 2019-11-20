# Feed-forward model, LM training
LOAD_PREFIX=ff_task_agnostic
MODE=task_agnostic
FINETUNE=no
#FINETUNE=yes
MDL=default
##TEST_SAMPLES=10

# Feed-forward model, multitask training
#LOAD_PREFIX=ff_multitask
#MODE=single_task
#FINETUNE=yes
#MDL=default
#TEST_SAMPLES=10

# Feed-forward model, meta training
#LOAD_PREFIX=ff_meta
#MODE=single_task
#FINETUNE=z
#MDL=default
##TEST_SAMPLES=10

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_lm_seq_trans
export DBG_MODE=no
export LAYERS=4
#export ZSIZE=128

#export LOAD=$LOAD
export MODE=$MODE
export FINETUNE=$FINETUNE
export MDL=$MDL
#export TEST_SAMPLES=$TEST_SAMPLES

#export CUDA_VISIBLE_DEVICES=0

for ts in 1 5 10 20  ; do  #25 30 40 50 ; do
	export TEST_SAMPLES=$ts
	for z in 128 ; do
		export ZSIZE=$z
		for run in `(seq 1 5)` ; do
			LOAD=${LOAD_PREFIX}_zsize${z}_run${run}
			export LOAD=${LOAD}

			./scripts/eval_synthetic_lm.sh

			sleep $[ ( $RANDOM % 10 ) + 1 ]s
		done
	done
done

#for ts in 5 10 20 ; do
#	export TEST_SAMPLES=$ts
#	for z in 32  ; do
#		export ZSIZE=$z
#		for run in 4 5  ; do
#			LOAD=${LOAD_PREFIX}_zsize${z}_run${run}
#			export LOAD=${LOAD}
#
#			./scripts/eval_synthetic_lm.sh
#		done
#	done
#done
