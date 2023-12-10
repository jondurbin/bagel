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
)
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
        default=0.015, metdata={"help": "warmup ratio"}
    )
    neftune_noise_alpha: Optional[int] = field(
        default=5, metdata={"help": "NEFTune noise alpha"}
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
    use_flash_attention_2: Optional[bool] = field(
        default=True,
        metadata={"help": "use flash-attn 2.*"},
    )
    use_fast_tokenizer: Optional[bool] = field(
        default=False,
        metadata={"help": "use the fast tokenizer"},
    )
    dataset: Optional[str] = field(
        default="bagel-dpo-v0.1.parquet",
        help={"help": "parquet dataset to use for fine-tuning"},
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


def train():
    parser = HfArgumentParser(ScriptArguments)
    script_args = parser.parse_args_into_dataclasses()[0]

    model = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
    )

    if script_args.ignore_bias_buffers:
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    model_ref = AutoModelForCausalLM.from_pretrained(
        script_args.model_name_or_path,
        use_flash_attention_2=script_args.use_flash_attention_2,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        script_args.model_name_or_path, use_fast=script_args.use_fast_tokenizer
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    full_dataset = Dataset.from_parquet(script_args.dataset).train_test_split(
        test_size=script_args.eval_dataset_size
    )
    train_dataset = full_dataset["train"]
    eval_dataset = full_dataset["test"]

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
    )

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
    )

    dpo_trainer.train()

    dpo_trainer.accelerator.wait_for_everyone()
    state_dict = dpo_trainer.accelerator.get_state_dict(dpo_trainer.deepspeed)
    unwrapped_model = dpo_trainer.accelerator.unwrap_model(dpo_trainer.deepspeed)
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
