#!/bin/bash

# How to start it without my conda etc.? 
# see how I can run module load X in a script
# Try running it with the full path to the poetry executable or the full path to the python executable.


poetry run --directory /home/kit/iti/zn2950/ws/src/master_thesis/legacy_code/ main evaluate --settings /home/kit/iti/zn2950/ws/src/master_thesis/legacy_code/settings/settings_llama_31.yml --stats-path /home/kit/iti/zn2950/ws/lsdf_home/results/llama_31_8b_instruct_stats_lsdf.yml --only-model "LLaMa 3.1 8B Instruct"
