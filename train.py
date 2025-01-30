import re
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import GRPOConfig, GRPOTrainer
from utils import format_reward_func, reward_func
from gsm8k import GSM8K


dataset = GSM8K(split='train', include_answer=False, include_reasoning=True, few_shot=True, num_shots=2, seed=None, cot=True, template="qa").dataset
eval_dataset = GSM8K(split='test', include_answer=False, include_reasoning=True, few_shot=True, num_shots=2, seed=None, cot=True, template="qa").dataset

model_name = "Qwen/Qwen2.5-Math-1.5B"
output_dir = f'outputs/GRPO/{model_name}'

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=f'GRPO-GSM8K-{model_name.split('/')[-1]}',
    learning_rate=5e-6,
    adam_beta1 = 0.9,
    adam_beta2 = 0.99,
    weight_decay = 0.1,
    warmup_ratio = 0.1,
    lr_scheduler_type='cosine',
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=6,
    max_prompt_length=512,
    max_completion_length=512,
    num_train_epochs=1,
    save_steps=100,
    max_grad_norm=0.1,
    report_to='wandb',
    log_on_each_node=False
    )


rank = 8
peft_config = LoraConfig(
    r=rank,
    lora_alpha=rank*2,
    target_modules=None,
    task_type="CAUSAL_LM",
    bias='none',
    lora_dropout=0.05,
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype='auto',
    attn_implementation="flash_attention_2",
    device_map='auto'
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        reward_func
        ],
    args=training_args,
    train_dataset=dataset,
    eval_dataset=eval_dataset,
    peft_config=peft_config
)

trainer.train()

model.save_pretrained(output_dir)
print(f"LoRA model and configuration saved to {output_dir}")