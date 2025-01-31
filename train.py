import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from utils import format_reward_func, correctness_reward_func, print_trainable_parameters
from gsm8k import GSM8K


dataset = GSM8K(split='train', include_answer=False, include_reasoning=True, few_shot=True, num_shots=2, seed=None, cot=True, template="qa").dataset.shuffle(seed=42)

model_name = "Qwen/Qwen2.5-Math-1.5B"
output_dir = f'outputs/GRPO/{model_name}'

training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=f'GRPO-GSM8K-{model_name.split('/')[-1]}',
    learning_rate=2e-5,
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    num_generations=6,
    max_prompt_length=256,
    max_completion_length=300,
    num_train_epochs=2,
    save_steps=100,
    max_grad_norm=0.1,
    report_to='wandb',
    log_on_each_node=False,
    # use_vllm=True,
    # vllm_device='auto',
)


rank = 16
peft_config = LoraConfig(
    r=rank,
    lora_alpha=rank*2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    bias='none',
    lora_dropout=0.05,
)


model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map='auto'
)

model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

trainer = GRPOTrainer(
    model=model,
    processing_class=tokenizer,
    reward_funcs=[
        format_reward_func,
        correctness_reward_func
        ],
    args=training_args,
    train_dataset=dataset,
)

trainer.train()

model.save_pretrained(output_dir)
print(f"LoRA model and configuration saved to {output_dir}")