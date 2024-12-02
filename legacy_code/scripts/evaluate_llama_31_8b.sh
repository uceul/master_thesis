#!/bin/bash

poetry run main evaluate --settings settings/settings_llama_31.yml --stats_path llama_31_8b_stats.yml --only-model "LLaMa 3.1 8B"
