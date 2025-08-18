# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import time
import torch.distributed as dist
import torch.nn as nn
import logging
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Mxfp4Config
from kernels import kernelize, Mode
from transformers.trainer_callback import TrainerCallback
import torch
from torch.profiler import ProfilerActivity, tensorboard_trace_handler, schedule

from trl import (
    ModelConfig as TrlModelConfig,
    ScriptArguments,
    SFTConfig as TrlSFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)

class TokenMetricsCallback(TrainerCallback):
    """
    A `TrainerCallback` that computes and logs the tokens per second and tokens per second per GPU.
    """

    def __init__(self):
        self.last_time = None
        self.last_num_tokens = 0

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None or "num_tokens" not in logs:
            return

        current_time = time.time()

        # On the first log, just record the state and return
        if self.last_time is None:
            self.last_time = current_time
            self.last_num_tokens = logs["num_tokens"]
            return

        time_delta = current_time - self.last_time
        tokens_delta = logs["num_tokens"] - self.last_num_tokens

        if time_delta > 0:
            tokens_per_sec = tokens_delta / time_delta
            tokens_per_sec_per_gpu = tokens_per_sec / dist.get_world_size()

            logs["tokens_per_sec"] = round(tokens_per_sec, 2)
            logs["tokens_per_sec_per_gpu"] = round(tokens_per_sec_per_gpu, 2)

        self.last_time = current_time
        self.last_num_tokens = logs["num_tokens"]


class MfuMetricsCallback(TrainerCallback):
    """
    A `TrainerCallback` that computes and logs the memory usage.
    """

    def __init__(self):
        # Source: https://github.com/pytorch/torchtitan/pull/1559/files#diff-871c1e4f538476256309628f6be55a8b6fa474e7e29fc95f5154e5095fdd6e3fR88
        self.last_time = None
        self.last_num_tokens = 0
        self.nparams = 0
        self.num_flops_per_token = 0

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        # `model` is the PeftModel, we need the underlying base model
        if hasattr(model, "model"):
            model = model.model
        config = model.config

        self.num_experts_per_tok = getattr(config, "num_experts_per_tok", 0)
        self.num_local_experts = getattr(config, "num_local_experts", 1)  # Non-moe models have 1 expert.
        self.num_hidden_layers = config.num_hidden_layers
        self.num_attention_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.seq_len = config.initial_context_length

        self.nparams, self.num_flops_per_token = self.get_nparams_and_flops(model)
        # https://github.com/stanford-cs336/spring2024-lectures/blob/main/lecture_02.py#L950
        self.theoretical_flops = 989.5 * 10 ** 12

    def get_nparams_and_flops(self, model: nn.Module) -> tuple[int, int]:
        """
        Adopted from llama4 implementation.
        """
        nparams_embedding = 0
        nparams_moe_router = 0
        nparams_shared_expert = 0
        nparams_experts = 0
        nparams_dense = 0

        for name, p in model.named_parameters():
            if "embedding" in name:
                nparams_embedding += p.numel()
                nparams_dense += p.numel()
            elif "moe.shared_expert" in name:
                nparams_shared_expert += p.numel()
            elif "moe.router" in name:
                nparams_moe_router += p.numel()
            elif "moe.experts" in name:
                nparams_experts += p.numel()
            else:
                nparams_dense += p.numel()

        nparams_sparse = nparams_moe_router + nparams_shared_expert + nparams_experts
        nparams = nparams_dense + nparams_sparse
        nparams_sparse_active = (
            nparams_moe_router
            + nparams_shared_expert
            + nparams_experts * self.num_experts_per_tok // self.num_local_experts
        )

        l, h, q, t = (
            self.num_hidden_layers,
            self.num_attention_heads,
            self.hidden_size // self.num_attention_heads,
            self.seq_len,
        )
        # Reasoning behind the factor of 12 for the self-attention part of the formula:
        # 1. each self-attention has 2 matmul in the forward and 4 in the backward (6)
        # 2. the flash attention does 1 more matmul recomputation in the backward
        #    but recomputation should not be counted in calculating MFU           (+0)
        # 3. each matmul performs 1 multiplication and 1 addition                 (*2)
        # 4. we follow the convention and do not account for sparsity in causal attention
        num_flops_per_token = 6 * (nparams_dense - nparams_embedding + nparams_sparse_active) + 12 * l * h * q * t

        return nparams, num_flops_per_token

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero or logs is None or "num_tokens" not in logs:
            return

        current_time = time.time()

        # On the first log, just record the state and return
        if self.last_time is None:
            self.last_time = current_time
            self.last_num_tokens = logs["num_tokens"]
            return

        time_delta = current_time - self.last_time
        tokens_delta = logs["num_tokens"] - self.last_num_tokens

        if time_delta > 0:
            tokens_per_sec = tokens_delta / time_delta
            tokens_per_sec_per_gpu = tokens_per_sec / dist.get_world_size()

            achieved_mfu = (tokens_per_sec_per_gpu * self.num_flops_per_token) / self.theoretical_flops
            logs["mfu"] = round(achieved_mfu * 100, 2)

        self.last_time = current_time
        self.last_num_tokens = logs["num_tokens"]

class ProfCallback(TrainerCallback):
    def __init__(self, prof):
        self.prof = prof

    def on_step_end(self, args, state, control, **kwargs):
        if dist.get_rank() == 0:
            self.prof.step()

@dataclass
class ModelConfig(TrlModelConfig):
    use_kernels: bool = field(default=False, metadata={"help": "Enable/disable kernels"})
    use_tp: bool = field(default=False, metadata={"help": "Alias for tp to handle --use_tp flag."})

@dataclass
class SFTConfig(TrlSFTConfig):
    wandb_project: str = field(default="oai-3outeille", metadata={"help": "Wandb project name."})
    wandb_entity: str = field(default="huggingface", metadata={"help": "Wandb entity name."})

def main(script_args, training_args, model_args):
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    quantization_config = Mxfp4Config(dequantize=True)
    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation,
        torch_dtype=model_args.torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        quantization_config=quantization_config,
        use_kernels=model_args.use_kernels,
    )

    if model_args.use_tp:
        model_kwargs["tp_plan"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path, **model_kwargs
    )
    model = kernelize(model, mode=Mode.TRAINING)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    # --------------
    # Load dataset
    # --------------
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, split=script_args.dataset_train_split)
    dataset = dataset.select(range(int(len(dataset) * 0.1)))

    # -------------
    # Train model
    # -------------
    training_args.report_to = "wandb"

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args) if model_args.use_peft else None,
        callbacks=[
            TokenMetricsCallback(),
            MfuMetricsCallback(),
        ]
    )

    with torch.profiler.profile(activities=[ProfilerActivity.CPU,
                                            ProfilerActivity.CUDA], 
                                schedule=schedule(skip_first=0, wait=5, warmup=2, active=2, repeat=1),
                                # - No skip (skip_first=0)
                                # - Iteration 1-5: Wait (wait=5)
                                # - Iteration 5-6: Warmup (warmup=2)
                                # - Iterations 7-8-9: Active profiling (active=3)
                                # - After iteration 9: Profiling complete (repeat=1), so it will not start a new cycle after the 6th iteration.
                                on_trace_ready=tensorboard_trace_handler(dir_name=f"./profiler", worker_name=f"worker_{dist.get_rank()}"),
                                profile_memory=True,
                                with_stack=True,
                                record_shapes=True,) as prof:

        # trainer.add_callback(ProfCallback(prof=prof))       
        trainer.train()
    
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )

    if "wandb" in training_args.report_to:
        os.environ["WANDB_PROJECT"] = training_args.wandb_project
        os.environ["WANDB_ENTITY"] = training_args.wandb_entity

    main(script_args, training_args, model_args)
