#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python api_server.py 2>&1 | tee log/output_$(date +%Y-%m-%d-%H-%M-%S).log