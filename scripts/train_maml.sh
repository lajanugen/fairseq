# Feed-forward model, LM training
#EXP_NAME=ff_taskagnostic
MODE=maml
MDL=default

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_lm_fix
export DBG_MODE=no
export LAYERS=2
export ZSIZE=128
#export NUMGRADS=1
#export NUMGRADS=25

#export EXP_NAME=$EXP_NAME
export MODE=$MODE
export MDL=$MDL
#export CUDA_VISIBLE_DEVICES=0

for numg in 1 3 5 ; do
   export NUMGRADS=$numg
   for ly in 1 2 4 ; do
      export LAYERS=$ly

      EXP_NAME=test_maml_layer${ly}_numgrads${numg}_encdim8  #ff_${md}_zsize${zsize}_run${run}
      export EXP_NAME=$EXP_NAME
      ./scripts/train_synthetic_lm_maml.sh
   done
done


