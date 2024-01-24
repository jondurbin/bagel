# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import copy
import json
import os
import re
import uuid
import shutil
from os.path import exists, join, isdir
from dataclasses import dataclass, field
from typing import Optional, Dict, Sequence, Any
import numpy as np
from tqdm import tqdm
import logging
import warnings
import bitsandbytes as bnb
import pandas as pd
import importlib
from packaging import version
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
import argparse
from transformers import (
    AddedToken,
    AutoTokenizer,
    AutoModelForCausalLM,
    set_seed,
    Seq2SeqTrainer,
    BitsAndBytesConfig,
    LlamaTokenizer,
)
from datasets import load_dataset, Dataset
import evaluate

from peft import (
    prepare_model_for_kbit_training,
    AutoPeftModelForCausalLM,
    LoraConfig,
    get_peft_model,
    PeftModel,
)
from peft.tuners.lora import LoraLayer
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.trainer_pt_utils import get_parameter_names
from accelerate import Accelerator


def is_ipex_available():
    def get_major_and_minor_from_version(full_version):
        return (
            str(version.parse(full_version).major)
            + "."
            + str(version.parse(full_version).minor)
        )

    _torch_version = importlib.metadata.version("torch")
    if importlib.util.find_spec("intel_extension_for_pytorch") is None:
        return False
    _ipex_version = "N/A"
    try:
        _ipex_version = importlib.metadata.version("intel_extension_for_pytorch")
    except importlib.metadata.PackageNotFoundError:
        return False
    torch_major_and_minor = get_major_and_minor_from_version(_torch_version)
    ipex_major_and_minor = get_major_and_minor_from_version(_ipex_version)
    if torch_major_and_minor != ipex_major_and_minor:
        warnings.warn(
            f"Intel Extension for PyTorch {ipex_major_and_minor} needs to work with PyTorch {ipex_major_and_minor}.*,"
            f" but PyTorch {_torch_version} is found. Please switch to the matching version and run again."
        )
        return False
    return True


if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True

logger = logging.getLogger(__name__)

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default="meta-llama/Llama-2-7b",
    )
    trust_remote_code: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."
        },
    )
    use_auth_token: Optional[bool] = field(
        default=False,
        metadata={"help": "Enables using Huggingface auth token from Git Credentials."},
    )


@dataclass
class DataArguments:
    eval_dataset_size: float = field(
        default=0.02, metadata={"help": "Size of validation dataset."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    model_max_len: int = field(
        default=4096,
        metadata={
            "help": "Maximum model length (input and output).  Sequences will be right padded (and possibly truncated)."
        },
    )
    skip_excess_length: bool = field(
        default=True, metadata={"help": "Purge dataset items that exceed model_max_len"}
    )
    dataset: str = field(
        default="instructions.jsonl",
        metadata={"help": "Which dataset to finetune on. See datamodule for options."},
    )
    dataset_format: Optional[str] = field(
        default="airoboros",
        metadata={
            "help": "Which dataset format is used. [alpaca|chip2|self-instruct|hh-rlhf|airoboros]"
        },
    )
    expand_conversations: bool = field(
        default=False,
        metadata={"help": "Expand all multi-turn conversations, use with care"},
    )
    include_sources: Optional[str] = field(
        default="ALL",
        metadata={
            "help": "Comma separated list of sources to include (source field in dataset)"
        },
    )


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    train_on_source: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to train on the input in addition to the target text."
        },
    )
    mmlu_split: Optional[str] = field(
        default="eval", metadata={"help": "The MMLU split to run on"}
    )
    mmlu_dataset: Optional[str] = field(
        default="mmlu-fs",
        metadata={
            "help": "MMLU dataset to use: options are `mmlu-zs` for zero-shot or `mmlu-fs` for few shot."
        },
    )
    do_mmlu_eval: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run the MMLU evaluation."}
    )
    max_mmlu_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, only evaluates on `max_mmlu_samples` of the MMMLU dataset."
        },
    )
    mmlu_source_max_len: int = field(
        default=2048, metadata={"help": "Maximum source sequence length for mmlu."}
    )
    full_finetune: bool = field(
        default=False, metadata={"help": "Finetune the entire model without adapters."}
    )
    adam8bit: bool = field(default=False, metadata={"help": "Use 8-bit adam."})
    double_quant: bool = field(
        default=True,
        metadata={
            "help": "Compress the quantization statistics through double quantization."
        },
    )
    quant_type: str = field(
        default="nf4",
        metadata={
            "help": "Quantization data type to use. Should be one of `fp4` or `nf4`."
        },
    )
    bits: int = field(default=4, metadata={"help": "How many bits to use."})
    lora_r: int = field(default=64, metadata={"help": "Lora R dimension."})
    lora_alpha: float = field(default=16, metadata={"help": " Lora alpha."})
    lora_dropout: float = field(default=0.0, metadata={"help": "Lora dropout."})
    max_memory_MB: int = field(default=80000, metadata={"help": "Free memory per gpu."})
    report_to: str = field(
        default="none",
        metadata={"help": "To use wandb or something else for reporting."},
    )
    use_fast_tokenizer: bool = field(
        default=False, metadata={"help": "Use fast tokenizer"}
    )
    pad_token: str = field(
        default=None, metadata={"help": "Custom pad token, e.g. for qwen"}
    )
    eos_token: str = field(
        default=None, metadata={"help": "Custom EOS token, e.g. for qwen"}
    )
    bos_token: str = field(
        default=None, metadata={"help": "Custom BOS token, e.g. for qwen"}
    )
    unk_token: str = field(
        default=None, metadata={"help": "Custom UNK token, e.g. for qwen"}
    )
    padding_side: str = field(
        default="right", metadata={"help": "tokenizer padding side"}
    )
    final_output_dir: str = field(
        default="./final",
        metadata={"help": "The final output directory, for completed model"},
    )
    output_dir: str = field(
        default="./output",
        metadata={"help": "The output (and intermediate) directory."},
    )
    optim: str = field(
        default="paged_adamw_32bit", metadata={"help": "The optimizer to be used"}
    )
    per_device_train_batch_size: int = field(
        default=1,
        metadata={
            "help": "The training batch size per GPU. Increase for better speed."
        },
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={"help": "The eval batch size per GPU. Increase for better speed."},
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={
            "help": "How many gradients to accumulate before to perform an optimizer step"
        },
    )
    num_train_epochs: int = field(
        default=3, metadata={"help": "Number of training epochs."}
    )
    weight_decay: float = field(
        default=0.0, metadata={"help": "The L2 weight decay rate of AdamW"}
    )  # use lora dropout instead for regularization if needed
    learning_rate: float = field(default=0.0002, metadata={"help": "The learning rate"})
    remove_unused_columns: bool = field(
        default=False,
        metadata={"help": "Removed unused columns. Needed to make this codebase work."},
    )
    max_grad_norm: float = field(
        default=0.3,
        metadata={
            "help": "Gradient clipping max norm. This is tuned and works well for all models tested."
        },
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing. You want to use this."},
    )
    do_train: bool = field(
        default=True,
        metadata={"help": "To train or not to train, that is the question?"},
    )
    lr_scheduler_type: str = field(
        default="constant",
        metadata={
            "help": "Learning rate schedule. Constant a bit better than cosine, and has advantage for analysis"
        },
    )
    warmup_ratio: float = field(
        default=0.03, metadata={"help": "Fraction of steps to do a warmup for"}
    )
    logging_steps: int = field(
        default=10,
        metadata={"help": "The frequency of update steps after which to log the loss"},
    )
    group_by_length: bool = field(
        default=False,
        metadata={
            "help": "Group sequences into batches with same length. Saves memory and speeds up training considerably."
        },
    )
    save_strategy: str = field(
        default="steps", metadata={"help": "When to save checkpoints"}
    )
    save_steps: int = field(default=250, metadata={"help": "How often to save a model"})
    save_total_limit: int = field(
        default=40,
        metadata={
            "help": "How many checkpoints to save before the oldest is overwritten"
        },
    )
    deepspeed: str = field(
        default=None, metadata={"help": "deepspeed configuration path"}
    )
    using_fsdp: bool = field(
        default=False,
        metadata={
            "help": "Flag indicating whether or not you are using FSDP (via accelerate)"
        },
    )
    max_shard_size: str = field(
        default="5GB",
        metadata={"help": "Max shard size when saving model after full finetune."},
    )
    save_quantized_base: bool = field(
        default=False, metadata={"help": "Optionally save the quantized base model"}
    )
    attn_implementation: str = field(
        default=None, metadata={"help": "Attention implementation."}
    )
    neftune_noise_alpha: int = field(
        default=5, metadata={"help": "NEFTune noise alpha value"}
    )
    layers_to_freeze: int = field(
        default=0,
        metadata={
            "help": "Number of layers to freeze, reduces VRAM, can produce worse results"
        },
    )
    layer_freeze_ratio: float = field(
        default=0.0,
        metadata={
            "help": "Ratio of layers to freeze, reduces VRAM, can produce worse results"
        },
    )
    nadam: bool = field(default=False, metadata={"help": "Use NAdam optimizer"})
    radam: bool = field(default=False, metadata={"help": "Use RAdam optimizer"})


def find_all_linear_names(args, model):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


class SavePeftModelCallback(transformers.TrainerCallback):
    def __init__(self, trainer, **_):
        self.trainer = trainer

    def save_model(self, args, state, kwargs):
        print("Saving PEFT checkpoint...")
        checkpoint_folder = os.path.join(
            args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
        )
        peft_model_path = os.path.join(checkpoint_folder, "adapter_model")

        if getattr(self.trainer, "deepspeed"):
            self.trainer.accelerator.wait_for_everyone()
            state_dict = self.trainer.accelerator.get_state_dict(self.trainer.deepspeed)
            unwrapped_model = self.trainer.accelerator.unwrap_model(
                self.trainer.deepspeed
            )
            if self.trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(
                    peft_model_path, state_dict=state_dict, safe_serialization=True
                )
            self.trainer.accelerator.wait_for_everyone()
        else:
            kwargs["model"].save_pretrained(peft_model_path, safe_serialization=True)

        pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
        if os.path.exists(pytorch_model_path):
            os.remove(pytorch_model_path)
        try:
            if os.path.exists(
                os.path.join(checkpoint_folder, f"global_step{state.global_step}")
            ):
                print(f"Cleaning up global_step{state.global_step}")
                shutil.rmtree(
                    os.path.join(checkpoint_folder, f"global_step{state.global_step}")
                )
        except Exception as exc:
            print(f"Failed to clean up global_step{state.global_step}: {exc}")

    def on_save(self, args, state, control, **kwargs):
        self.save_model(args, state, kwargs)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        def touch(fname, times=None):
            with open(fname, "a"):
                os.utime(fname, times)

        self.save_model(args, state, kwargs)
        touch(join(args.output_dir, "completed"))


def get_accelerate_model(args, checkpoint_dir):
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    if is_ipex_available() and torch.xpu.is_available():
        n_gpus = torch.xpu.device_count()
    max_memory = f"{args.max_memory_MB}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"

    # if we are in a distributed setting, we need to set the device map and max memory per device
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    if args.full_finetune:
        assert args.bits in [16, 32]

    # Tokenizer...
    extra_tokens = {}
    for key in ("pad_token", "eos_token", "bos_token", "unk_token"):
        value = getattr(args, key, None)
        if value:
            extra_tokens[key] = value
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        use_fast=args.use_fast_tokenizer,
        padding_side=args.padding_side,
        tokenizer_type="llama"
        if "llama" in args.model_name_or_path
        else None,  # Needed for HF name change
        trust_remote_code=args.trust_remote_code,
        **extra_tokens,
    )
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.pad_token = tokenizer.unk_token

    # Ensure the model has the correct token IDs (qwen!!!)
    extra_model_args = {}
    for key in ("pad_token", "eos_token", "bos_token", "unk_token"):
        value = getattr(args, key, None)
        if value:
            extra_model_args[f"{key}_id"] = getattr(tokenizer, f"{key}_id")
    if "qwen" in args.model_name_or_path:
        extra_model_args["bf16"] = True
        extra_model_args["use_flash_attn"] = True

    # Model...
    print(f"loading base model {args.model_name_or_path}...")
    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )
    bnb_config = None
    if not args.full_finetune and args.bits in (4, 8):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        cache_dir=args.cache_dir,
        load_in_4bit=args.bits == 4,
        load_in_8bit=args.bits == 8,
        quantization_config=bnb_config,
        torch_dtype=(
            torch.float32
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        device_map=None if args.deepspeed else device_map,
        max_memory=None if args.deepspeed else max_memory,
        **extra_model_args,
    )
    if compute_dtype == torch.float16 and args.bits == 4:
        if torch.cuda.is_bf16_supported():
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    if compute_dtype == torch.float16 and (
        is_ipex_available() and torch.xpu.is_available()
    ):
        compute_dtype = torch.bfloat16
        print("Intel XPU does not support float16 yet, so switching to bfloat16")

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float32 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    # Freeze embeddings layer.
    model.model.embed_tokens.weight.requires_grad_(False)

    # Add the stupid chatml tokens.
    if "qwen" not in args.model_name_or_path:
        tokenizer.add_tokens(
            [
                transformers.AddedToken("<|im_start|>", special=True, normalized=False),
                transformers.AddedToken("<|im_end|>", special=True, normalized=False),
            ]
        )

        # Resize the tokenizer to be divisible by 64 for better performance.
        if len(tokenizer) % 64 != 0:
            for idx in range((len(tokenizer) // 64 + 1) * 64 - len(tokenizer)):
                tokenizer.add_tokens(
                    [
                        transformers.AddedToken(
                            f"<|special_{idx}|>", special=True, normalized=False
                        )
                    ]
                )

    # Resize token embeddings, if necessary, to accomodate fast tokenizer with added tokens.
    if "qwen" not in args.model_name_or_path:
        num_new_tokens = len(tokenizer) - len(model.get_input_embeddings().weight.data)
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings_data = model.get_input_embeddings().weight.data
            output_embeddings_data = model.get_output_embeddings().weight.data
            input_embeddings_data[-num_new_tokens:] = 0.0
            output_embeddings_data[-num_new_tokens:] = 0.0
            model.model.embed_tokens.weight.requires_grad_(True)

    if not args.full_finetune and args.bits in (8, 4):
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )

    if args.gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
        if "norm" in name:
            module = module.to(torch.bfloat16 if args.bf16 else torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)

    if not args.full_finetune:
        if checkpoint_dir is not None:
            print("Loading adapters from checkpoint.")
            model = PeftModel.from_pretrained(
                model, join(checkpoint_dir, "adapter_model"), is_trainable=True
            )
        else:
            print("adding LoRA modules...")
            modules = find_all_linear_names(args, model)
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            model.enable_input_require_grads()
            model = get_peft_model(model, config)
    if args.using_fsdp:
        accelerator = Accelerator()
        model = accelerator.prepare_model(model)
    return model, tokenizer


def print_trainable_parameters(args, model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


@dataclass
class DataCollatorForCausalLM(object):
    tokenizer: transformers.PreTrainedTokenizer
    model_max_len: int
    train_on_source: bool
    predict_with_generate: bool

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        # Extract elements
        sources = [example["input"] for example in instances]
        targets = [example["output"] for example in instances]
        # Tokenize
        tokenized_sources_with_prompt = self.tokenizer(
            sources,
            max_length=self.model_max_len,
            truncation=True,
            add_special_tokens=True,
        )
        tokenized_targets = self.tokenizer(
            targets,
            max_length=self.model_max_len,
            truncation=True,
            add_special_tokens=False,
        )
        # Build the input and labels for causal LM
        input_ids = []
        labels = []
        for tokenized_source, tokenized_target in zip(
            tokenized_sources_with_prompt["input_ids"], tokenized_targets["input_ids"]
        ):
            truncated_target = False
            if len(tokenized_source) + len(tokenized_target) >= self.model_max_len:
                if len(tokenized_source) <= 512:
                    tokenized_target = tokenized_target[
                        0 : self.model_max_len - len(tokenized_source)
                    ]
                    truncated_target = True
                elif len(tokenized_target) <= 512:
                    tokenized_source = tokenized_source[
                        0 : self.model_max_len - len(tokenized_target)
                    ]
                else:
                    tokenized_source = tokenized_source[0 : int(self.model_max_len / 2)]
                    tokenized_target = tokenized_target[0 : int(self.model_max_len / 2)]
                    truncated_target = True

            if not self.predict_with_generate:
                target_inputs = tokenized_source + tokenized_target
                if not truncated_target:
                    target_inputs.append(self.tokenizer.eos_token_id)
                input_ids.append(torch.tensor(target_inputs))
                if not self.train_on_source:
                    target_labels = copy.deepcopy(tokenized_target)
                    if not truncated_target:
                        target_labels.append(self.tokenizer.eos_token_id)
                    labels.append(
                        torch.tensor(
                            [IGNORE_INDEX for _ in range(len(tokenized_source))]
                            + target_labels
                        )
                    )
                else:
                    if not truncated_target:
                        labels.append(
                            torch.tensor(
                                copy.deepcopy(
                                    tokenized_source
                                    + tokenized_target
                                    + [self.tokenizer.eos_token_id]
                                )
                            )
                        )
                    else:
                        labels.append(
                            torch.tensor(
                                copy.deepcopy(tokenized_source + tokenized_target)
                            )
                        )
            else:
                input_ids.append(torch.tensor(tokenized_source))
        # Apply padding
        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = (
            pad_sequence(labels, batch_first=True, padding_value=IGNORE_INDEX)
            if not self.predict_with_generate
            else None
        )
        data_dict = {
            "input_ids": input_ids,
            "attention_mask": input_ids.ne(self.tokenizer.pad_token_id),
        }
        if labels is not None:
            data_dict["labels"] = labels
        return data_dict


def extract_unnatural_instructions_data(examples, extract_reformulations=False):
    out = {
        "input": [],
        "output": [],
    }
    for example_instances in examples["instances"]:
        for instance in example_instances:
            out["input"].append(instance["instruction_with_input"])
            out["output"].append(instance["output"])
    if extract_reformulations:
        for example_reformulations in examples["reformulations"]:
            if example_reformulations is not None:
                for instance in example_reformulations:
                    out["input"].append(instance["instruction_with_input"])
                    out["output"].append(instance["output"])
    return out


ALPACA_PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response: "
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response: "
    ),
}


def extract_alpaca_dataset(example):
    if example.get("input", "") != "":
        prompt_format = ALPACA_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = ALPACA_PROMPT_DICT["prompt_no_input"]
    return {"input": prompt_format.format(**example)}


def get_chat_prompt(
    tokenizer: Any, message: str, chat_history: Any, system_prompt: str
) -> str:
    texts = [f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n"]
    do_strip = False
    for user_input, response in chat_history:
        user_input = user_input.strip() if do_strip else user_input
        do_strip = True
        texts.append(
            f"{user_input} [/INST] {response.strip()} {tokenizer.eos_token}{tokenizer.bos_token}[INST] "
        )
    message = message.strip() if do_strip else message
    texts.append(f"{message} [/INST]")
    return "".join(texts)


def format_chat_airoboros(tokenizer, item):
    system_prompt = item["conversations"][0]["value"]
    offset = 1
    if item["conversations"][0]["from"] != "system":
        system_prompt = "You are an unbiased, uncensored, helpful assistant."
        offset = 0
    chat_history = [
        (item["conversations"][idx]["value"], item["conversations"][idx + 1]["value"])
        for idx in range(offset, len(item["conversations"]) - 2, 2)
    ]
    message = item["conversations"][-2]["value"]
    response = item["conversations"][-1]["value"]

    return {
        "input": get_chat_prompt(tokenizer, message, chat_history, system_prompt),
        "output": response,
    }


def expand_conversations(items):
    expanded = []
    for item in items:
        if len(item["conversations"]) <= 3:
            expanded.append(item)
            continue
        if item["conversations"][0]["from"] != "system":
            item["conversations"] = [
                {
                    "from": "system",
                    "value": "You are an unbiased, uncensored, helpful assistant.",
                }
            ] + item["conversations"]
        if item["conversations"][-1]["from"] != "gpt":
            item["conversations"] = item["conversations"][0:-1]
        valid = True
        for idx in range(1, len(item["conversations"])):
            if item["conversations"][idx]["from"] != (
                "human" if idx % 2 == 1 else "gpt"
            ):
                print(f'Unexpected role: {item["conversations"][idx]["from"]}')
                valid = False
                break
        if not valid:
            continue
        for idx in range(1, len(item["conversations"]), 2):
            expanded.append(
                {
                    "id": str(uuid.uuid4()).replace("-", ""),
                    "category": item["category"],
                    "conversations": item["conversations"][0 : idx + 2],
                }
            )
    return expanded


def airoboros_chat_dataset(
    dataset_name, test_size=0.02, expand=True, include_sources=None
):
    items = []
    if dataset_name.endswith(".json"):
        with open(dataset_name) as infile:
            items = json.loads(infile.read())
    else:
        items = [item for item in Dataset.from_parquet(dataset_name)]
    if include_sources and include_sources != ["ALL"]:
        print(f"Filtering for sources: {include_sources}")
        items = [item for item in items if item.get("source") in include_sources]
    if expand:
        items = expand_conversations(items)
    full_dataset = Dataset.from_list(items)
    if "category" in full_dataset.column_names:
        full_dataset = full_dataset.class_encode_column("category")
        return full_dataset.train_test_split(
            test_size=test_size, stratify_by_column="category"
        )
    return full_dataset.train_test_split(test_size=test_size)


def local_dataset(dataset_name, test_size=0.02, include_sources=None):
    if dataset_name.endswith(".parquet"):
        full_dataset = Dataset.from_parquet(path_or_paths=dataset_name)
    elif dataset_name.endswith(".json") or dataset_name.endswith(".jsonl"):
        full_dataset = Dataset.from_json(path_or_paths=dataset_name)
    elif dataset_name.endswith(".csv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name))
    elif dataset_name.endswith(".tsv"):
        full_dataset = Dataset.from_pandas(pd.read_csv(dataset_name, delimiter="\t"))
    else:
        raise ValueError(f"Unsupported dataset format: {dataset_name}")
    if include_sources and include_sources != ["ALL"]:
        print(f"Filtering for sources: {include_sources}")
        full_dataset = full_dataset.filter(lambda x: x["source"] in include_sources)
    if "category" in full_dataset.column_names:
        full_dataset = full_dataset.class_encode_column("category")
        return full_dataset.train_test_split(
            test_size=test_size, stratify_by_column="category"
        )
    elif "source" in full_dataset.column_names:
        try:
            full_dataset = full_dataset.class_encode_column("source")
        except Exception:
            ...
        return full_dataset.train_test_split(
            test_size=test_size, stratify_by_column="source"
        )
    return full_dataset.train_test_split(test_size=test_size)


def make_data_module(tokenizer: transformers.PreTrainedTokenizer, args) -> Dict:
    """
    Make dataset and collator for supervised fine-tuning.
    Datasets are expected to have the following columns: { `input`, `output` }

    Available datasets to be selected with `dataset` argument:
        - alpaca, 52002 examples
        - alpaca cleaned, 51942 examples
        - chip2 (OIG), 210289 examples
        - self-instruct, 82612 examples
        - hh-rlhf (Anthropic), 160800 examples
        - longform, 23.7k examples
        - oasst1 (OpenAssistant) primary message tree only, 9,846 examples

    Coming soon:
        - unnatural instructions core, 66010 examples
        - unnatural instructions full, 240670 examples
        - alpaca-gpt4, 52002 examples
        - unnatural-instructions-gpt4, 9000 examples
        - supernatural-instructions, 69624 examples (same as paper with 100 ex/task more can be used)
        - flan (FLAN v2), up to 20M examples available
        - vicuna

    """

    def load_data(dataset_name):
        if dataset_name == "alpaca":
            return load_dataset("tatsu-lab/alpaca")
        elif dataset_name == "alpaca-clean":
            return load_dataset("yahma/alpaca-cleaned")
        elif dataset_name == "chip2":
            return load_dataset("laion/OIG", data_files="unified_chip2.jsonl")
        elif dataset_name == "self-instruct":
            return load_dataset("yizhongw/self_instruct", name="self_instruct")
        elif dataset_name == "hh-rlhf":
            return load_dataset("Anthropic/hh-rlhf")
        elif dataset_name == "longform":
            return load_dataset("akoksal/LongForm")
        elif dataset_name == "oasst1":
            return load_dataset("timdettmers/openassistant-guanaco")
        elif dataset_name == "vicuna":
            raise NotImplementedError("Vicuna data was not released.")
        else:
            if os.path.exists(dataset_name):
                try:
                    args.dataset_format = (
                        args.dataset_format if args.dataset_format else "input-output"
                    )
                    full_dataset = (
                        airoboros_chat_dataset(
                            dataset_name,
                            args.eval_dataset_size,
                            args.expand_conversations,
                            args.include_sources.split(","),
                        )
                        if args.dataset_format == "airoboros_chat"
                        else local_dataset(
                            dataset_name,
                            args.eval_dataset_size,
                            include_sources=args.include_sources.split(","),
                        )
                    )
                    return full_dataset
                except Exception:
                    raise ValueError(f"Error loading dataset from {dataset_name}")
            else:
                raise NotImplementedError(
                    f"Dataset {dataset_name} not implemented yet."
                )

    def format_dataset(dataset, dataset_format):
        if (
            dataset_format == "alpaca"
            or dataset_format == "alpaca-clean"
            or (dataset_format is None and args.dataset in ["alpaca", "alpaca-clean"])
        ):
            dataset = dataset.map(
                extract_alpaca_dataset, remove_columns=["instruction"]
            )
        elif dataset_format == "chip2" or (
            dataset_format is None and args.dataset == "chip2"
        ):
            dataset = dataset.map(
                lambda x: {
                    "input": x["text"].split("\n<bot>: ")[0].replace("<human>: ", ""),
                    "output": x["text"].split("\n<bot>: ")[1],
                }
            )
        elif dataset_format == "self-instruct" or (
            dataset_format is None and args.dataset == "self-instruct"
        ):
            for old, new in [["prompt", "input"], ["completion", "output"]]:
                dataset = dataset.rename_column(old, new)
        elif dataset_format == "hh-rlhf" or (
            dataset_format is None and args.dataset == "hh-rlhf"
        ):
            dataset = dataset.map(lambda x: {"input": "", "output": x["chosen"]})
        elif dataset_format == "oasst1" or (
            dataset_format is None and args.dataset == "oasst1"
        ):
            dataset = dataset.map(
                lambda x: {
                    "input": "",
                    "output": x["text"],
                }
            )
        elif dataset_format == "airoboros":

            def _format_airoboros(instruction):
                in_ = None
                if instruction.get("skip_prompt_formatting"):
                    in_ = instruction["instruction"].strip() + "\n"
                else:
                    in_ = "\n".join(
                        [
                            (instruction.get("system") or "A chat.").strip(),
                            f"USER: {instruction['instruction'].strip()}",
                        ]
                    )
                    if in_.endswith("PLAINFORMAT"):
                        in_ = re.sub(r"\s+PLAINFORMAT$", "", in_, re.DOTALL)
                        in_ += " PLAINFORMAT"
                    in_ = "\n".join([in_.strip(), "ASSISTANT: "])
                return {
                    "input": in_,
                    "output": instruction["response"].strip() + "\n",
                }

            dataset = dataset.map(_format_airoboros)
        elif dataset_format == "airoboros_chat":
            dataset = dataset.map(lambda x: format_chat_airoboros(tokenizer, x))
        elif dataset_format == "input-output":
            # leave as is
            pass
        # Remove unused columns.
        dataset = dataset.remove_columns(
            [
                col
                for col in dataset.column_names["train"]
                if col not in ["input", "output"]
            ]
        )
        return dataset

    # Load dataset.
    dataset = load_data(args.dataset)
    dataset = format_dataset(dataset, args.dataset_format)

    # Split train/eval, reduce size
    if args.do_eval or args.do_predict:
        if "eval" in dataset:
            eval_dataset = dataset["eval"]
        elif "test" in dataset:
            eval_dataset = dataset["test"]
        else:
            print(
                "Splitting train dataset in train and validation according to `eval_dataset_size`"
            )
            if "category" in dataset["train"].column_names:
                dataset["train"] = dataset["train"].class_encode_column("category")
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size,
                    stratify_by_column="category",
                    seed=args.seed,
                )
            else:
                dataset = dataset["train"].train_test_split(
                    test_size=args.eval_dataset_size, shuffle=True, seed=args.seed
                )
            eval_dataset = dataset["test"]
        if (
            args.max_eval_samples is not None
            and len(eval_dataset) > args.max_eval_samples
        ):
            eval_dataset = eval_dataset.select(range(args.max_eval_samples))
        if args.group_by_length:
            eval_dataset = eval_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )
    if args.do_train:
        train_dataset = dataset["train"]
        if (
            args.max_train_samples is not None
            and len(train_dataset) > args.max_train_samples
        ):
            train_dataset = train_dataset.select(range(args.max_train_samples))
        if args.group_by_length:
            train_dataset = train_dataset.map(
                lambda x: {"length": len(x["input"]) + len(x["output"])}
            )

    # Remove any training data that exceeds the max length.
    if args.skip_excess_length:

        def _get_data_length(item):
            prompt = f"{tokenizer.bos_token}{item['input']}{item['output']}{tokenizer.eos_token}"
            return len(
                tokenizer(
                    prompt,
                    max_length=args.model_max_len + 1,
                    truncation=True,
                    add_special_tokens=False,
                ).input_ids
            )

        train_dataset = train_dataset.filter(
            lambda x: _get_data_length(x) < args.model_max_len - 10
        )

    data_collator = DataCollatorForCausalLM(
        tokenizer=tokenizer,
        model_max_len=args.model_max_len,
        train_on_source=args.train_on_source,
        predict_with_generate=args.predict_with_generate,
    )
    return dict(
        train_dataset=train_dataset if args.do_train else None,
        eval_dataset=eval_dataset if args.do_eval else None,
        predict_dataset=eval_dataset if args.do_predict else None,
        data_collator=data_collator,
    )


def get_last_checkpoint(checkpoint_dir):
    if isdir(checkpoint_dir):
        is_completed = exists(join(checkpoint_dir, "completed"))
        if is_completed:
            return None, True  # already finished
        max_step = 0
        for filename in os.listdir(checkpoint_dir):
            if isdir(join(checkpoint_dir, filename)) and filename.startswith(
                "checkpoint"
            ):
                max_step = max(max_step, int(filename.replace("checkpoint-", "")))
        if max_step == 0:
            return None, is_completed  # training started, but no checkpoint
        checkpoint_dir = join(checkpoint_dir, f"checkpoint-{max_step}")
        print(f"Found a previous checkpoint at: {checkpoint_dir}")
        return checkpoint_dir, is_completed  # checkpoint found!
    return None, False  # first training


def train():
    hfparser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    (
        model_args,
        data_args,
        training_args,
        extra_args,
    ) = hfparser.parse_args_into_dataclasses(return_remaining_strings=True)
    args = argparse.Namespace(
        **vars(model_args), **vars(data_args), **vars(training_args)
    )
    print(args)

    checkpoint_dir, completed_training = get_last_checkpoint(args.output_dir)
    if completed_training:
        print("Detected that training was already completed!")

    model, tokenizer = get_accelerate_model(args, checkpoint_dir)

    # Freeze layers, if specified.
    layers_to_freeze = 0
    if args.layers_to_freeze:
        layers_to_freeze = args.layers_to_freeze
    elif args.layer_freeze_ratio:
        layers_to_freeze = int(len(model.model.layers) * args.layer_freeze_ratio)
    if layers_to_freeze:
        for param in model.parameters():
            param.requires_grad = False
        for param in model.lm_head.parameters():
            param.requires_grad = True
        for param in model.model.layers[layers_to_freeze:].parameters():
            param.requires_grad = True
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:

            def make_inputs_require_grad(module, in_, output):
                output.requires_grad_(True)

            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    model.config.use_cache = False
    print("loaded model")
    set_seed(args.seed)

    data_module = make_data_module(tokenizer=tokenizer, args=args)

    # Support RAdam and NAdam.
    if args.radam or args.nadam:
        decay_parameters = get_parameter_names(model, [torch.nn.LayerNorm])
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in model.named_parameters() if n in decay_parameters
                ],
                "weight_decay": args.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters() if n not in decay_parameters
                ],
                "weight_decay": 0.0,
            },
        ]
        optim = "RAdam" if args.radam else "NAdam"
        training_args.optimizer = getattr(torch.optim, optim)(
            optimizer_grouped_parameters,
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

    trainer = Seq2SeqTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    # Callbacks
    if not args.full_finetune:
        trainer.add_callback(SavePeftModelCallback(trainer))
    if args.do_mmlu_eval:
        if args.mmlu_dataset == "mmlu-zs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "eval": "data/mmlu/zero_shot_mmlu_val.json",
                    "test": "data/mmlu/zero_shot_mmlu_test.json",
                },
            )
            mmlu_dataset = mmlu_dataset.remove_columns("subject")
        # MMLU Five-shot (Eval/Test only)
        elif args.mmlu_dataset == "mmlu" or args.mmlu_dataset == "mmlu-fs":
            mmlu_dataset = load_dataset(
                "json",
                data_files={
                    "eval": "data/mmlu/five_shot_mmlu_val.json",
                    "test": "data/mmlu/five_shot_mmlu_test.json",
                },
            )
            # mmlu_dataset = mmlu_dataset.remove_columns('subject')
        mmlu_dataset = mmlu_dataset[args.mmlu_split]
        if args.max_mmlu_samples is not None:
            mmlu_dataset = mmlu_dataset.select(range(args.max_mmlu_samples))
        abcd_idx = [
            tokenizer("A", add_special_tokens=False).input_ids[0],
            tokenizer("B", add_special_tokens=False).input_ids[0],
            tokenizer("C", add_special_tokens=False).input_ids[0],
            tokenizer("D", add_special_tokens=False).input_ids[0],
        ]
        accuracy = evaluate.load("accuracy")

        class MMLUEvalCallback(transformers.TrainerCallback):
            def on_evaluate(self, args, state, control, model, **kwargs):
                data_loader = trainer.get_eval_dataloader(mmlu_dataset)
                model_max_len = trainer.data_collator.model_max_len
                trainer.data_collator.model_max_len = args.mmlu_source_max_len
                trainer.model.eval()
                preds, refs = [], []
                loss_mmlu = 0
                for batch in tqdm(data_loader, total=len(data_loader)):
                    (loss, logits, labels) = trainer.prediction_step(
                        trainer.model,
                        batch,
                        prediction_loss_only=False,
                    )
                    # There are two tokens, the output, and eos token.
                    for i, logit in enumerate(logits):
                        label_non_zero_id = (batch["labels"][i] != -100).nonzero()[0][0]
                        logit_abcd = logit[label_non_zero_id - 1][abcd_idx]
                        preds.append(torch.argmax(logit_abcd).item())
                    labels = labels[labels != IGNORE_INDEX].view(-1, 2)[:, 0]
                    refs += [abcd_idx.index(label) for label in labels.tolist()]
                    loss_mmlu += loss.item()
                # Extract results by subject.
                results = {"mmlu_loss": loss_mmlu / len(data_loader)}
                subject = mmlu_dataset["subject"]
                subjects = {s: {"refs": [], "preds": []} for s in set(subject)}
                for s, p, r in zip(subject, preds, refs):
                    subjects[s]["preds"].append(p)
                    subjects[s]["refs"].append(r)
                subject_scores = []
                for subject in subjects:
                    subject_score = accuracy.compute(
                        references=subjects[subject]["refs"],
                        predictions=subjects[subject]["preds"],
                    )["accuracy"]
                    results[
                        f"mmlu_{args.mmlu_split}_accuracy_{subject}"
                    ] = subject_score
                    subject_scores.append(subject_score)
                results[f"mmlu_{args.mmlu_split}_accuracy"] = np.mean(subject_scores)
                trainer.log(results)
                trainer.data_collator.model_max_len = model_max_len

        trainer.add_callback(MMLUEvalCallback)

    # Verifying the datatypes and parameter counts before training.
    if not args.deepspeed:
        print_trainable_parameters(args, model)
    if not args.full_finetune:
        dtypes = {}
        for _, p in model.named_parameters():
            dtype = p.dtype
            if dtype not in dtypes:
                dtypes[dtype] = 0
            dtypes[dtype] += p.numel()
        total = 0
        for k, v in dtypes.items():
            total += v
        for k, v in dtypes.items():
            print(k, v, v / total)

    all_metrics = {"run_name": args.run_name}
    # Training
    if args.do_train:
        logger.info("*** Train ***")
        # Note: `resume_from_checkpoint` not supported for adapter checkpoints by HF.
        # Currently adapter checkpoint is reloaded as expected but optimizer/scheduler states are not.
        train_result = trainer.train()
        metrics = train_result.metrics
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        all_metrics.update(metrics)
    # Evaluation
    if args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(metric_key_prefix="eval")
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        all_metrics.update(metrics)
    # Prediction
    if args.do_predict:
        logger.info("*** Predict ***")
        prediction_output = trainer.predict(
            test_dataset=data_module["predict_dataset"], metric_key_prefix="predict"
        )
        prediction_metrics = prediction_output.metrics
        predictions = prediction_output.predictions
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        predictions = tokenizer.batch_decode(
            predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "predictions.jsonl"), "w") as fout:
            for i, example in enumerate(data_module["predict_dataset"]):
                example["prediction_with_input"] = predictions[i].strip()
                example["prediction"] = (
                    predictions[i].replace(example["input"], "").strip()
                )
                fout.write(json.dumps(example) + "\n")
        print(prediction_metrics)
        trainer.log_metrics("predict", prediction_metrics)
        trainer.save_metrics("predict", prediction_metrics)
        all_metrics.update(prediction_metrics)

    if args.do_train or args.do_eval or args.do_predict:
        os.makedirs(args.output_dir, exist_ok=True)
        with open(os.path.join(args.output_dir, "metrics.json"), "w") as fout:
            fout.write(json.dumps(all_metrics))

    # Safely save final full-tune model.
    if args.full_finetune:
        trainer.accelerator.wait_for_everyone()
        state_dict = None
        unwrapped_model = None
        if args.deepspeed:
            state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
        else:
            state_dict = trainer.accelerator.get_state_dict(trainer.model)
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        if trainer.accelerator.is_main_process:
            unwrapped_model.save_pretrained(
                args.final_output_dir,
                state_dict=state_dict,
                max_shard_size=args.max_shard_size,
            )
            with open(os.path.join(args.final_output_dir, "config.json")) as infile:
                config = json.loads(infile.read())
            config["_name_or_path"] = os.path.basename(args.final_output_dir)
            with open(
                os.path.join(args.final_output_dir, "config.json"), "w"
            ) as outfile:
                outfile.write(json.dumps(config, indent=2))
            tokenizer.save_pretrained(args.final_output_dir)
        trainer.accelerator.wait_for_everyone()
    else:
        if args.deepspeed:
            trainer.accelerator.wait_for_everyone()
            state_dict = trainer.accelerator.get_state_dict(trainer.deepspeed)
            unwrapped_model = trainer.accelerator.unwrap_model(trainer.deepspeed)
            if trainer.accelerator.is_main_process:
                unwrapped_model.save_pretrained(
                    args.final_output_dir,
                    safe_serialization=True,
                    state_dict=state_dict,
                )
            trainer.accelerator.wait_for_everyone()
        else:
            trainer.accelerator.wait_for_everyone()
            if trainer.accelerator.is_main_process:
                trainer.model.save_pretrained(
                    args.final_output_dir, safe_serialization=True
                )
            trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
