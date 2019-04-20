#!/usr/bin/env bash

# This is a script that will train all of the models for scene graph classification and then evaluate them.
#export CUDA_VISIBLE_DEVICES=$1
    
# echo "TRAINING MOTIFNET"
# python -m models.train_rels -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5 \
#     -p 100 -hidden_dim 512 -pooling_dim 4096 -ngpu 2 -ckpt /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/motifnet_objdet/vg-faster-rcnn.tar -use_bias -save_dir random/ -nepoch 5 -lr 1e-3

echo "EVALING MOTIFNET"
# python -m models.eval_rels -m sgcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5 \
#     -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/temp/vgrel-0.tar -nepoch 50 -use_bias -cache motifnet_sgcls
# python -m models.eval_rels -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5 \
#     -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/temp/vgrel-0.tar -nepoch 50 -use_bias -cache motifnet_predcls
python -m models._visualize -m predcls -model motifnet -order leftright -nl_obj 2 -nl_edge 4 -b 1 -clip 5 \
    -p 100 -hidden_dim 512 -pooling_dim 4096 -lr 1e-3 -ngpu 1 -test -ckpt /scratch/cluster/ankgarg/gqa/temp_abhinav/checkpoints/temp/vgrel-0.tar -nepoch 50 -use_bias -cache motifnet_visualize