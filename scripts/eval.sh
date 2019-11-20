export CUDA_VISIBLE_DEVICES=0
RUN=./scripts/run_lm.sh

# Task agnostic model, no fine-tuning
# LOAD=taskag_lr1e-5/checkpoint_last.pt
# $RUN -m task_agnostic --mdl default --max-epoch 1 -i $LOAD --runmode eval --no-training

# Task agnostic model, full fine-tuning
# LOAD=taskag_lr1e-5/checkpoint_last.pt
# $RUN -m task_agnostic --mdl default --max-epoch 50 -i $LOAD --runmode eval --zsize 768 --lr 1e-4

# Meta model, adapter tuning
# LOAD=ff_meta_v6_adapter2_10upd_lr1e-5/checkpoint_last.pt
# $RUN -m single_task --mdl default --max-epoch 50 -i $LOAD --runmode eval -z --lr 1e-2 --ztype adapters --zsize 7684

# Meta model, encoder representation tuning
# LOAD=meta_encoder_10upd_lr1e-5/checkpoint_last.pt
# $RUN -m single_task --mdl default --max-epoch 50 -i $LOAD --runmode eval -z --lr 1e-2 --ztype encoder --zsize 768
