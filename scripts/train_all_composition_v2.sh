## Task agnostic
#EXP_NAME=multitask
#MODE=multitask

# Feed-forward model, LM training
#EXP_NAME=ff_taskagnostic
#MODE=task_agnostic

## DEFALUTS ##
export CKPT_DIR=/checkpoint/annl/transfer_learn_composite_v4
export DBG_MODE=no
export LAYERS=4
#export ZSIZE=128
#export NUMGRADS=100
export NUMGRADS=25

export KSHOTSAMPLE=0
export MAXEPOCH=100
export STAGE=-1

#export EXP_NAME=$EXP_NAME
#export MODE=$MODE
export MDL=default
#export CUDA_VISIBLE_DEVICES=0
export LOG_LOSS=1

for op in div+replace-ith-next+swap mul+replace-ith+reverse div+replace-ith+shift add+replace-ith-jth+swap add+replace-ith-jth+reverse mod+replace-ith-next+swap add+replace-ith-next+swap mul+replace-ith-jth+swap ; do 

   export TRAIN_FILE=tasks-${op}.train.txt

#   export KSHOTSAMPLE=0
#   export MAXEPOCH=100
#   export MODE=task_agnostic
#   export ZSIZE=128
#   export EXP_NAME=${MODE}_${op}
#
#   ./scripts/train_composition.sh

#   export MODE=maml
#   export ZSIZE=128
#   for ng in 5 ; do # 10 25 ; do
#	export NUMGRADS=${ng} 
#	for lr in 1e-5 1e-4 1e-3 1e-2 ; do
#		export LR=$lr
#		for zlr in 1e-5 1e-4 1e-3 1e-2 1e-1 ; do
#			export ZLR=$zlr
#			export EXP_NAME=maml_${op}_lr${lr}_zlr${zlr}_numgrad${ng}
#
#			./scripts/train_composition.sh
#		done
#		sleep $[ ( $RANDOM % 5 ) + 1 ]s
#	done
#   done

   for md in meta_randmask ; do # meta multitask; do # meta meta_randmask multitask meta_fixstage ; do  
      export MODE=$md

      for zsize in 128 64 32 16 ; do
	export ZSIZE=$zsize

#	export MAXEPOCH=500
#	for ts in 1 5 10 20 ; do
#		export KSHOTSAMPLE=$ts
#		for lr in 1e-3 2e-3 ; do
#			export LR=$lr
#			for zlr in 5e-2 1e-1 ; do
#				export ZLR=$zlr
#				export EXP_NAME=${md}_${op}_zsize${zsize}_${ts}shot_lr${lr}_zlr${zlr}
#		
#				./scripts/train_composition.sh
#			done
#		done
#		sleep $[ ( $RANDOM % 5 ) + 1 ]s
#	done

	for zlr in 5e-2 8e-2 1e-1 ; do
		export ZLR=$zlr
		export EXP_NAME=${md}_${op}_zsize${zsize}_zlr${zlr}
	
		./scripts/train_composition.sh
	done

#	export EXP_NAME=${md}_${op}_zsize${zsize}
#	./scripts/train_composition.sh
      done
      
#      sleep $[ ( $RANDOM % 5 ) + 1 ]s
   done

done

