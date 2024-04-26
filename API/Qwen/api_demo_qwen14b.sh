#!/bin/bash

python openai_api.py 2>&1 | tee log/output_$(date +%Y-%m-%d-%H-%M-%S).log