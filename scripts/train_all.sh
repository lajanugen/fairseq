# Training with different models and training algorithms
# The 4 examples below show how a feedforward/SNAIL model can be trained with
# multitask training or meta training algorithms

# # 1) Feedforward model, multitask training
# EXP_NAME_PREFIX=ff_multi_zsize32
# MODE=multitask
# MDL=default
# 
# # 2) Feedforward model, meta training
# EXP_NAME_PREFIX=ff_meta_zsize32
# MODE=meta_zeroinit
# MDL=default
# ZLR=1e-2
# NUMGRADS=25
#
# # 3) SNAIL model, multitask training
# # Train with at most 4 training context sequences
# EXP_NAME_PREFIX=snail_ex80_zsize32
# MODE=snail
# MDL=snail
# TEST_SAMPLES=80
# 
# # 4) SNAIL model, meta training
# EXP_NAME_PREFIX=snail_meta_ex80
# MODE=meta
# MDL=snail
# TEST_SAMPLES=80
#
# # 5) Feedforward model, maml training
# EXP_NAME_PREFIX=ff_maml
# MODE=maml
# MDL=maml
# ZLR=1e-6
# NUMGRADS=5
#
 # 6) Feedforward model, maml meta training
 EXP_NAME_PREFIX=ff_maml_meta
 MODE=maml_meta
 MDL=default
 ZLR=1e-1
 NUMGRADS=25
# To train a feedforward model with multitask training, use 1)
# Eg: 

#EXP_NAME=ff_multi
#MODE=multitask
#MDL=default

## DEFALUTS ##
CKPT_DIR=/checkpoint/annl/transfer_learn
DBG_MODE=no
LAYERS=4
#ZSIZE=128
ZSIZE=32
LR=1e-3
ZINNERLR=1e-5

export CKPT_DIR=$CKPT_DIR
#export EXP_NAME=$EXP_NAME
export MODE=$MODE
export MDL=$MDL
export TEST_SAMPLES=$TEST_SAMPLES
export DBG_MODE=$DBG_MODE
export LAYERS=$LAYERS
export ZSIZE=$ZSIZE
export NUMGRADS=$NUMGRADS
export LR=$LR
export ZLR=$ZLR
export ZINNERLR=$ZINNERLR
#export CUDA_VISIBLE_DEVICES=0

for runid in `(seq 1 5)`  ; do
  suffix="run$runid" 
  export EXP_NAME="$EXP_NAME_PREFIX""_$suffix"
  ./scripts/train.sh
done

