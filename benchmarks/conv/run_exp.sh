#!/bin/bash
ITERS=${1:-'10'}

CUDA_VISIBLE_DEVICES=3 python conv.py $ITERS --cost_model random  --search_eps 1.0   
CUDA_VISIBLE_DEVICES=3 python conv.py $ITERS --cost_model random  --search_eps 0.05  
CUDA_VISIBLE_DEVICES=3 python conv.py $ITERS --cost_model random  --search_eps 0.5   
CUDA_VISIBLE_DEVICES=3 python conv.py $ITERS --cost_model XGBoost --search_eps 0.05  
CUDA_VISIBLE_DEVICES=3 python conv.py $ITERS --cost_model XGBoost --search_eps 0.5   
wait
python conv_torch.py

