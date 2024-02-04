from dataclasses import dataclass, field
from typing import Optional
import os
import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    BitsAndBytesConfig,
    AddedToken,
)
from peft import prepare_model_for_kbit_training, PeftModel
from peft.tuners.lora import LoraLayer
from trl import DPOTrainer


@dataclass
class ScriptArguments:
    """
    The arguments for the DPO training script.
    """

    beta: Optional[float] = field(
        default=0.1, metadata={"help": "the beta parameter for DPO loss"}
    )
    model_name_or_path: Optional[str] = field(
        default="mistralai/mistral-7b-v0.1", metadata={"help": "the model name"}
    )
    adapter_path: Optional[str] = field(
        default=None, metadata={"help": "path to adapter model when using (q)lora"}
    )
    four_bit: Optional[bool] = field(
        default=False, metadata={"help": "use 4-bit quantization"}
    )
    learning_rate: Optional[float] = field(
        default=5e-7, metadata={"help": "optimizer learning rate"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=4, metadata={"help": "batch size per device"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=1, metadata={"help": "the number of gradient accumulation steps"}
    )
    warmup_ratio: Optional[float] = field(
        default=0.015, metadata={"help": "warmup ratio"}
    )
    neftune_noise_alpha: Optional[int] = field(
        default=5, metadata={"help": "NEFTune noise alpha"}
    )
    max_length: Optional[int] = field(
        default=512, metadata={"help": "max length of each sample"}
    )
    max_prompt_length: Optional[int] = field(
        default=128, metadata={"help": "max length of each sample's prompt"}
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "Only used for encoder decoder model. Max target of each sample's prompt"
        },
    )
    label_pad_token_id: Optional[int] = field(
        default=-100, metadata={"help": "label for non response tokens"}
    )
    num_train_epochs: Optional[int] = field(
        default=3, metadata={"help": "max number of epochs"}
    )
    sanity_check: Optional[bool] = field(
        default=True, metadata={"help": "only train on 1000 samples"}
    )
    report_to: Optional[str] = field(
        default="wandb",
        metadata={
            "help": "integration to report results and logs to, e.g. wandb, tensorboard, etc.",
        },
    )
    ignore_bias_buffers: Optional[bool] = field(
        default=False,
        metadata={
            "help": "fix for DDP issues with LM bias/mask buffers - invalid scalar type,`inplace operation. See"
            "https://github.com/huggingface/transformers/issues/22482#issuecomment-1595790992"
        },
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Whether to use gradient checkpointing or no"}
    )
    gradient_checkpointing_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "key word arguments to be passed along `torch.utils.checkpoint.checkpoint` method - e.g. `use_reentrant=False`"
        },
    )
    attn_implementation: Optional[str] = field(
        default=None,
        metadata={"help": "attention implementation to use"},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "use the fast tokenizer"},
    )
    dataset: Optional[str] = field(
        default="bagel-dpo-v0.1.parquet",
        metadata={"help": "parquet dataset to use for fine-tuning"},
    )
    eval_steps: Optional[int] = field(
        default=5, metadata={"help": "number of steps between evaluations"}
    )
    eval_dataset_size: Optional[float] = field(
        default=0.03, metadata={"help": "eval dataset size"}
    )
    workdir: Optional[str] = field(
        default="workdir",
        metadata={"help": "path to save intermediate checkpoints and such"},
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "path to save final fine-tuned model"}
    )
    deepspeed: Optional[str] = field(
        default=None, metadata={"help": "optional path to deepspeed configuration file"}
    )
    save_steps: Optional[int] = field(
        default=25, metadata={"help": "number of steps between checkpoint saves"}
    )
    save_total_limit: Optional[int] = field(
        default=3, metadata={"help": "maximum number of checkpoints to save"}
    )
    max_memory: int = field(default=80000, metadata={"help": "max vram per gpu"})
    add_chatml_tokens: bool = field(default=False, metadata={"help": "add chatml tokens, if using a base model without them"})


def train():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    training_args = TrainingArguments(
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_train_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        remove_unused_columns=False,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        learning_rate=script_args.learning_rate,
        evaluation_strategy="steps",
        logging_first_step=True,
        logging_steps=1,
        eval_steps=script_args.eval_steps,
        output_dir=script_args.workdir,
        optim="rmsprop",
        warmup_ratio=script_args.warmup_ratio,
        report_to=script_args.report_to,
        bf16=True,
        gradient_checkpointing=script_args.gradient_checkpointing,
        neftune_noise_alpha=script_args.neftune_noise_alpha,
        deepspeed=script_args.deepspeed,
        save_strategy="steps",
        save_steps=script_args.save_steps,
        save_total_limit=script_args.save_total_limit,
        ddp_find_unused_parameters=False,
    )

    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
    max_memory = f"{script_args.max_memory}MB"
    max_memory = {i: max_memory for i in range(n_gpus)}
    device_map = "auto"
    if os.environ.get("LOCAL_RANK") is not None:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_map = {"": local_rank}
        max_memory = {"": max_memory[local_rank]}

    bnb_config = None
    if script_args.four_bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        low_cpu_mem_usage=not script_args.deepspeed,
        quantization_config=bnb_config,
        trust_remote_code=True,
        attn_implementation=script_args.attn_implementation,
        torch_dtype=torch.bfloat16,
        max_memory=None if script_args.deepspeed else max_memory,
        device_map=None if script_args.deepspeed else device_map,
    )
    model.config.use_cache = False

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    # Only instantiate a reference model if we aren't using PEFT.
    model_ref = None
    if not script_args.adapter_path:
        model_ref = AutoModelForCausalLM.from_pretrained(
            script_args.model_name_or_path,
            low_cpu_mem_usage=not script_args.deepspeed,
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=script_args.attn_implementation,
            torch_dtype=torch.bfloat16,
            max_memory=None if script_args.deepspeed else max_memory,
            device_map=None if script_args.deepspeed else device_map,
        )
        model_ref.config.use_cache = False

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, use_fast=script_args.use_fast_tokenizer, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if script_args.add_chatml_tokens:
        tokenizer.add_tokens(
            [
                AddedToken("<|im_start|>", special=True, normalized=False),
                AddedToken("<|im_end|>", special=True, normalized=False),
            ]
        )
        if len(tokenizer) % 64 != 0:
            tokens = [
                AddedToken(
                    f"<|special_{idx}|>", special=True, normalized=False
                )
                for idx in range((len(tokenizer) // 64 + 1) * 64 - len(tokenizer))
            ]
            tokenizer.add_tokens(tokens)
        num_new_tokens = len(tokenizer) - len(model.get_input_embeddings().weight.data)
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings_data = model.get_input_embeddings().weight.data
            output_embeddings_data = model.get_output_embeddings().weight.data
            input_embeddings_data[-num_new_tokens:] = 0.0
            output_embeddings_data[-num_new_tokens:] = 0.0

    if script_args.four_bit:
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=script_args.gradient_checkpointing
        )

    for mod in (model, model_ref):
        if not mod:
            continue
        for name, module in mod.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if "norm" in name:
                module = module.to(torch.bfloat16)
            if "lm_head" in name or "embed" in name or "output" in name:
                if hasattr(module, "weight"):
                    if module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    if script_args.gradient_checkpointing and hasattr(
        model, "gradient_checkpointing_enable"
    ):
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if script_args.adapter_path:
        model = PeftModel.from_pretrained(
            model, script_args.adapter_path, is_trainable=True, adapter_name="_train_adapter"
        )
        model.load_adapter(script_args.adapter_path, adapter_name="_ref_adapter")

    full_dataset = Dataset.from_parquet(script_args.dataset).train_test_split(
        test_size=script_args.eval_dataset_size
    )
    train_dataset = full_dataset["train"]
    eval_dataset = full_dataset["test"]

    dpo_trainer = DPOTrainer(
        model,
        model_ref,
        args=training_args,
        beta=script_args.beta,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=script_args.max_length,
        max_target_length=script_args.max_target_length,
        max_prompt_length=script_args.max_prompt_length,
        generate_during_eval=False,
        model_adapter_name="_train_adapter" if script_args.adapter_path else None,
        ref_adapter_name="_ref_adapter" if script_args.adapter_path else None,
    )

    dpo_trainer.train()

    dpo_trainer.accelerator.wait_for_everyone()
    state_dict = None
    unwrapped_model = None
    if script_args.deepspeed:
        state_dict = dpo_trainer.accelerator.get_state_dict(dpo_trainer.deepspeed)
        unwrapped_model = dpo_trainer.accelerator.unwrap_model(dpo_trainer.deepspeed)
    else:
        state_dict = dpo_trainer.accelerator.get_state_dict(dpo_trainer.model)
        unwrapped_model = dpo_trainer.accelerator.unwrap_model(dpo_trainer.model)
    if dpo_trainer.accelerator.is_main_process:
        unwrapped_model.save_pretrained(
            script_args.output_dir, state_dict=state_dict, max_shard_size="4GB"
        )
        with open(os.path.join(script_args.output_dir, "config.json")) as infile:
            config = json.loads(infile.read())
        config["_name_or_path"] = os.path.basename(script_args.output_dir)
        with open(os.path.join(script_args.output_dir, "config.json"), "w") as outfile:
            outfile.write(json.dumps(config, indent=2))
        tokenizer.save_pretrained(script_args.output_dir)
    dpo_trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()
