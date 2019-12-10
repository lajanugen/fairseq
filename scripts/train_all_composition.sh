## Task agnostic
#EXP_NAME=multitask
#MODE=multitask

# Feed-forward model, LM training
#EXP_NAME=ff_taskagnostic
#MODE=task_agnostic

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_composite_v2
export DBG_MODE=no
export LAYERS=4
#export ZSIZE=128
#export NUMGRADS=10
export NUMGRADS=25

#export EXP_NAME=$EXP_NAME
#export MODE=$MODE
export MDL=default
#export CUDA_VISIBLE_DEVICES=0



for op in add reverse replace-ith-jth ; do
   export TRAIN_FILE=tasks-${op}.train.txt

   export MODE=task_agnostic
   export ZSIZE=128
   export EXP_NAME=${MODE}_${op}

   ./scripts/train_composition.sh

   for md in multitask meta ; do
      for zsize in 128 64 32 16 ; do
	export ZSIZE=$zsize
	export MODE=$md

	export EXP_NAME=${md}_${op}_zsize${zsize}
	./scripts/train_composition.sh
      done
   done

done
