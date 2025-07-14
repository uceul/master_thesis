# Information Extraction From MoF Synthesis Descriptions using LLMs

This project contains the code of my Master's thesis project which deals with information extraction from scientific literature using LLMs.
The project uses the code of a former Master's student as a foundation which can be found here: https://github.com/aimat-lab/llm-data-extraction

Two main parts: 
- enhanced evaluation framework located in legacy_code
- torchtune configs and scripts located in tunes

For installation i would recommend two separate environments.

- Framework installation:

Install Poetry for your user
First create a python env (I used python 11) and load a cuda module (e.g. CUDA 12.4)
Then run poetry install in the legacy_code dir
Afterwards run pip install accelerate==0.33.0
(And maybe update transformer package)
=> Then use poetry run main <command> for interacting with framework

- Torchtune can be installed (I used python 11 again) using pip install torch torchvision torchao
- Then run a run_tune.sh script in the model's tune subdir.

For later reference, pip list outputs for both models are located in legacy_code/*pip_list.txt

Happy tuning!
