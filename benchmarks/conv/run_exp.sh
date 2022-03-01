#!/bin/bash
ITERS=${1-'10'}

CUDA_VISIBLE_DEVICES=0 python conv.py $ITERS --cost_model random  --search_eps 0.05  >& log0.txt &
CUDA_VISIBLE_DEVICES=1 python conv.py $ITERS --cost_model random  --search_eps 0.5   >& log1.txt &
CUDA_VISIBLE_DEVICES=2 python conv.py $ITERS --cost_model XGBoost --search_eps 0.05  >& log2.txt &
CUDA_VISIBLE_DEVICES=3 python conv.py $ITERS --cost_model XGBoost --search_eps 0.5   >& log3.txt &
wait
python conv_torch.py

