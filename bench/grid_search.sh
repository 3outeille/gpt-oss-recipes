#!/bin/bash

#TODO:
# make sure sft_tp.py and sft.py have
# 1) mfu computation
# 2) memory usage
# 3) max for 300 steps

# Zero3 + lora
python create_config.py --zero3 --lora
sbatch template.slurm configs/zero3_lora.yaml

# EP + lora
python create_config.py --tp --lora
sbatch template.slurm configs/tp_lora.yaml

# Zero3 + lora + megablocks
python create_config.py --zero3 --lora --megablocks
sbatch template.slurm configs/zero3_lora_megablocks.yaml

# EP + lora + megablocks
python create_config.py --tp --lora --megablocks
sbatch template.slurm configs/tp_lora_megablocks.yaml

# Zero3 + lora + megablocks + flash
python create_config.py --zero3 --lora --megablocks --flash
sbatch template.slurm configs/zero3_lora_megablocks_flash.yaml

# EP + lora + megablocks + flash
python create_config.py --tp --lora --megablocks --flash
sbatch template.slurm configs/tp_lora_megablocks_flash.yaml