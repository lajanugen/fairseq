# Feed-forward model, LM training
#LOAD_PREFIX=ff_task_agnostic
#MODE=task_agnostic
##FINETUNE=no
#FINETUNE=yes
#MDL=default
##TEST_SAMPLES=10

# Feed-forward model, multitask training
#LOAD_PREFIX=ff_multitask
#MODE=single_task
#FINETUNE=yes
#MDL=default
##TEST_SAMPLES=10

# Feed-forward model, meta training
LOAD_PREFIX=ff_meta
MODE=single_task
FINETUNE=z
MDL=default
#TEST_SAMPLES=10

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_lm_v2
export DBG_MODE=no
export LAYERS=4
#export ZSIZE=128

#export LOAD=$LOAD
export MODE=$MODE
export FINETUNE=$FINETUNE
export MDL=$MDL
#export TEST_SAMPLES=$TEST_SAMPLES

for ts in 1 5 10 20 25 30 40 50 ; do
	export TEST_SAMPLES=$ts
	for z in 128 64 32 16 8 ; do
		export ZSIZE=$z
		for run in 1 2 3 4 5 ; do
			LOAD=${LOAD_PREFIX}_zsize${z}_run${run}
			export LOAD=${LOAD}

			./scripts/eval_synthetic_lm_v2.sh
		done
	done
done

