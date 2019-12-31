## Task agnostic
#EXP_NAME=multitask
#MODE=multitask

# Feed-forward model, LM training
#EXP_NAME=ff_taskagnostic
#MODE=task_agnostic

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_composite_grid
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


export MODE=task_agnostic
export ZSIZE=128
export EXP_NAME=sample68_${MODE}

#./scripts/train_composition_grid.sh

for md in meta_randmask ; do # meta multitask ; do  # meta_fixstage ; do  
   for zsize in 128 ; do # 64 32 16 ; do
     export ZSIZE=$zsize
     export MODE=$md

     export EXP_NAME=sample68_${md}_zsize${zsize}
     ./scripts/train_composition_grid.sh
   done
   
done

	
