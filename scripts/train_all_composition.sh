## Task agnostic
#EXP_NAME=multitask
#MODE=multitask

# Feed-forward model, LM training
#EXP_NAME=ff_taskagnostic
#MODE=task_agnostic

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_composite_v3
export DBG_MODE=no
export LAYERS=4
#export ZSIZE=128
#export NUMGRADS=100
export NUMGRADS=25

#export EXP_NAME=$EXP_NAME
#export MODE=$MODE
export MDL=default
#export CUDA_VISIBLE_DEVICES=0
export LOG_LOSS=1

OLDIFS=$IFS
IFS=','

for testop in mod,0 div,0 mul,0 add,0 ; do # replace-ith-jth,1 replace-ith,1 replace-ith-next,1 reverse,2 swap,2 shift,2 sort,2 ; do  # mod-noadd,0 div-noadd,0 mul-noadd,0 ; do # 
   set -- $testop
   op=$1
   export STAGE=$2

   export TRAIN_FILE=tasks-${op}.train.txt

#   export MODE=task_agnostic
#   export ZSIZE=128
#   export EXP_NAME=${MODE}_${op}
#
#   ./scripts/train_composition.sh

   for md in meta_randmask ; do # meta multitask meta_fixstage ; do  
      for zsize in 128 64 32 16 ; do
	export ZSIZE=$zsize
	export MODE=$md

	export EXP_NAME=${md}_${op}_zsize${zsize}
	./scripts/train_composition.sh
      done
      
#      sleep $[ ( $RANDOM % 5 ) + 1 ]s
   done

done

IFS=$OLDIFS
	
