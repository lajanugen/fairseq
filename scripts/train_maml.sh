# Feed-forward model, LM training
#EXP_NAME=ff_taskagnostic
MODE=maml
MDL=default

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_lm_seq_trans
export TASKS_FILE=/checkpoint/annl/transfer_learn_lm_maml/tasks.txt
export VOCAB_SIZE=12
export SEQ_LEN=5

#export CKPT_DIR=/checkpoint/annl/transfer_learn_lm_grid
#export TASKS_FILE=/checkpoint/annl/transfer_learn_lm_grid/chunk_2000_gridworld-gen_len12.pkl
#export VOCAB_SIZE=200
#export SEQ_LEN=21

export DBG_MODE=no
export LAYERS=4
export ZSIZE=128
#export NUMGRADS=1
#export NUMGRADS=25

#export EXP_NAME=$EXP_NAME
export MODE=$MODE
export MDL=$MDL
#export CUDA_VISIBLE_DEVICES=0

export NUM_TRAIN_TASKS=500

#for numg in 5  ; do
#   export NUMGRADS=$numg
#   for lr in 1e-6 1e-5 1e-4 1e-3  ; do
#      export LR=$lr
#      for zlr in 1e-6 1e-5 1e-4 1e-3 1e-2 1e-1 ; do
#         export ZLR=$zlr
#
#	 EXP_NAME=test_numgrads${numg}_lr${lr}_zlr${zlr}_tasks${NUM_TRAIN_TASKS}_gpu1
#	 export EXP_NAME=$EXP_NAME
#
#      	 ./scripts/train_synthetic_lm_maml.sh
#      done
#   done
#done


for numg in 10 25  ; do
   export NUMGRADS=$numg
   for lr in 1e-3  ; do
      export LR=$lr
      for zlr in 1e-6  ; do
         export ZLR=$zlr

	 for runid in 1 2 3 4 5 ; do
   	    EXP_NAME=layer4_encdim128_numgrads${numg}_lr${lr}_zlr${zlr}_tasks500_gpu1_run${runid}  #grid_maml_numgrads${numg}_lr${lr}_zlr${zlr}_run${runid}
	    export EXP_NAME=$EXP_NAME

      	    ./scripts/train_synthetic_lm_maml.sh
	 done
      done
   done
done


