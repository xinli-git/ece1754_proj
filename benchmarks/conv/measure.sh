#!/bin/bash

CUDA_DEV=${1:-'2'}

export CUDA_VISIBLE_DEVICES=$CUDA_DEV

python conv_results_analysis.py 10000 --cost_model random --search_eps 0.05
python conv_results_analysis.py 10000 --cost_model random --search_eps 0.5
python conv_results_analysis.py 10000 --cost_model random --search_eps 1.0
python conv_results_analysis.py 10000 --cost_model lstm --search_eps 0.05
python conv_results_analysis.py 10000 --cost_model lstm --search_eps 0.5
python conv_results_analysis.py 10000 --cost_model lstm --search_eps 1.0
python conv_results_analysis.py 10000 --cost_model XGBoost --search_eps 0.05
python conv_results_analysis.py 10000 --cost_model XGBoost --search_eps 0.5

