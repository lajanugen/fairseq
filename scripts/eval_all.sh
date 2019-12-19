# The examples below show how a feedforward/SNAIL model can be adapted to new 
# tasks by either no-finetuning, full-finetuning or fine-tuning only the task
# embedding (z)

# 1) Full-finetuning a multi-task trained model
# LOAD_PREFIX=ff_multi
# MODE=single_task
# MDL=default
# FINETUNE=yes
# TEST_SAMPLES=80
# 
# 2) Fine-tune only z in a meta-trained model
# LOAD_PREFIX=ff_meta
# MODE=single_task
# MDL=default
# FINETUNE=z
# TEST_SAMPLES=80
# 
# 3) Evaluate a multi-task trained SNAIL model without any fine-tuning
# LOAD_PREFIX=snail_ex80_zsize32
# MODE=snail
# MDL=snail
# FINETUNE=no
# TEST_SAMPLES=80
# 
# 4) Meta trained SNAIL + fine-tuning only z
# LOAD_PREFIX=snail_meta_ex40
# MODE=single_task
# MDL=snail
# FINETUNE=z
# TEST_SAMPLES=40
# 
# # 5) Feedforward model, maml training
 LOAD_PREFIX=ff_maml
 MODE=maml
 MDL=maml
 FINETUNE=yes
 TEST_SAMPLES=80
 export ZLR=1e-6
 export NUMGRADS=5
#
# # 6) Feedforward model, maml meta training
# LOAD_PREFIX=ff_maml_meta
# MODE=maml_single_task
# MDL=default
# FINETUNE=z
# TEST_SAMPLES=4

export ZINNERLR=1e-3

# 
# To full-finetune a multi-task trained model, use 1)
# Eg: 

#LOAD=ff_multi
#MODE=single_task
#MDL=default
#FINETUNE=yes
#TEST_SAMPLES=4

## DEFALUTS ##
CKPT_DIR=/checkpoint/annl/transfer_learn
DBG_MODE=no
LAYERS=4
#ZSIZE=128
ZSIZE=32

export CKPT_DIR=$CKPT_DIR
#export LOAD=$LOAD
export MODE=$MODE
export MDL=$MDL
export FINETUNE=$FINETUNE
export TEST_SAMPLES=$TEST_SAMPLES
export DBG_MODE=$DBG_MODE
export LAYERS=$LAYERS
export ZSIZE=$ZSIZE
#export CUDA_VISIBLE_DEVICES=0

for run in run1 run2 run3 run4 run5; do
	LOAD="$LOAD_PREFIX""_$run"
	export LOAD=$LOAD
	./scripts/eval.sh
done
