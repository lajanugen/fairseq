# Feed-forward model, LM training
./scripts/run_lm.sh -m task_agnostic --mdl default -f --max-epoch 50 -i ff_taskagnostic/checkpoint10.pt --eval --no-training
