export CUDA_VISIBLE_DEVICES=0
RUN=./scripts/run_lm.sh 

# Meta model, task embedding = encoder representation
# EXP_NAME=meta_encoder_10upd_lr1e-5
# $RUN -e $EXP_NAME -m meta --mdl default ---ztype encoder -zsize 768 --tb --max-epoch 1 -i openwebtext.gpt/model.pt --lr 1e-5 --zlr 1e-2 --numgrads 10

# Meta model, task embedding = parameters in adapter layers
# EXP_NAME=meta_adapter2_10upd_lr1e-5
# $RUN -e $EXP_NAME -m meta --mdl default --ztype adapters --zsize 7684 --tb --max-epoch 1 -i openwebtext.gpt/model.pt --lr 1e-5 --zlr 1e-2 --numgrads 10

# Task agnostic model
# EXP_NAME=taskag_lr1e-5
# $RUN -e $EXP_NAME -m task_agnostic --mdl default --tb --max-epoch 5 -i openwebtext.gpt/model.pt --lr 1e-5

# SNAIL
# EXP_NAME=snail
# $RUN -e $EXP_NAME -m task_agnostic --mdl snail --tb --max-epoch 1 -i openwebtext.gpt/model.pt --lr 1e-7
