#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
#export CUDA_VISIBLE_DEVICES=$1
    
echo "TRAINING MOTIFNET"
python -m models.train_rels -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -ngpu 1 -ckpt /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/motifnet_objdet/vg-faster-rcnn.tar -use_bias $@
    
# -save_dir /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/motifnet_sgcls/ -nepoch 50 -lr 1e-3