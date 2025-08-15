#!/bin/bash

#TODO:
# make sure sft_tp.py and sft.py have
# 1) mfu computation
# 2) memory usage
# 3) max for 300 steps

# Zero3 + Peft
python create_config.py --zero3 --peft

# EP + Peft
python create_config.py --tp --peft

# Zero3 + Peft + megablocks
python create_config.py --zero3 --peft --megablocks

# EP + Peft + megablocks
python create_config.py --tp --peft --megablocks

# Zero3 + Peft + megablocks + flash
python create_config.py --zero3 --peft --megablocks --flash

# EP + Peft + megablocks + flash
python create_config.py --tp --peft --megablocks --flash