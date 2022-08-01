#!/bin/bash
export TVM_HOME=//home/lixin39/workspace/ece1754_tvm/tvm/
export PYTHONPATH=$TVM_HOME/python:${PYTHONPATH}
pip install tornado psutil 'xgboost<1.6.0' cloudpickle
