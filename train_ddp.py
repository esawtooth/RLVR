import argparse
import os
import torch
import torch.distributed as dist
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import GRPOConfig, GRPOTrainer
from utils import (
    format_reward_func_qa, correctness_reward_func_qa,
    format_reward_func_code, correctness_reward_func_code,
    print_trainable_parameters
)
from gsm8k import GSM8K

# ----- Distributed Setup -----
def init_distributed():
    # Initialize the distributed process group using NCCL (works for ROCm as well if built accordingly)
    dist.init_process_group(backend="nccl")
    # Get the local rank from the environment variable (set automatically by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank

# Call distributed initialization if running in a distributed setup.
local_rank = 0
if "LOCAL_RANK" in os.environ:
    local_rank = init_distributed()

# ----- Argument Parsing -----
def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune a model for Reasoning on GSM8K with RLVR and GRPO."
    )
    parser.add_argument('--format', type=str, default='qa', choices=['qa', 'code'])
    parser.add_argument('--num_shots', type=int, default=2)
    parser.add_argument('--model_name', type=str, default='Qwen/Qwen2.5-Math-1.5B')
    return parser.parse_args()

args = parse_args()
print(args)

# ----- Load the Dataset -----
dataset = GSM8K(
    split='train',
    include_answer=False,
    include_reasoning=True,
    few_shot=True,
    num_shots=args.num_shots,
    seed=None,
    cot=True,
    template=args.format
).dataset.shuffle(seed=42)

# ----- Set Output Directory -----
model_name = args.model_name
output_dir = f'outputs/GRPO/{args.format}/{model_name}'

# ----- Training Configuration (GRPO) -----
training_args = GRPOConfig(
    output_dir=output_dir,
    run_name=f'GRPO-GSM8K-{args.format}-{model_name.split("/")[-1]}',
    learning_rate=2e-5,
    logging_steps=1,
    bf16=True,
    per_device_train_batch_size=6,
    gradient_accumulation_steps=3,
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

# ----- LoRA Configuration -----
rank = 16
peft_config = LoraConfig(
    r=rank,
    lora_alpha=rank * 2,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj"],
    task_type="CAUSAL_LM",
    bias='none',
    lora_dropout=0.05,
)

# ----- Load Pre-trained Model on the Correct GPU -----
# Instead of using device_map='auto', we explicitly map the model to the GPU corresponding to local_rank.
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map={"": local_rank}  # Forces the model to load on the local GPU
)

# Apply LoRA to the model.
model = get_peft_model(model, peft_config)
print_trainable_parameters(model)

# If running in distributed mode, wrap the model with DistributedDataParallel.
if dist.is_initialized():
    from torch.nn.parallel import DistributedDataParallel as DDP
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

# ----- Load Tokenizer -----
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token
model.module.config.pad_token_id = tokenizer.pad_token_id if hasattr(model, 'module') else model.config.pad_token_id

# ----- Define Reward Functions -----
rewards_funcs = []
if args.format == 'qa':
    rewards_funcs = [format_reward_func_qa, correctness_reward_func_qa]
elif args.format == 'code':
    rewards_funcs = [format_reward_func_code, correctness_reward_func_code]

# ----- Initialize the GRPO Trainer -----
trainer = GRPOTrainer(
    model=model.module if hasattr(model, "module") else model,
    processing_class=tokenizer,
    reward_funcs=rewards_funcs,
    args=training_args,
    train_dataset=dataset,
)

# ----- Start Training -----
trainer.train()

# ----- Save the Fine-Tuned Model -----
# Only the main process should save the model.
if local_rank == 0:
    model_to_save = model.module if hasattr(model, "module") else model
    model_to_save.save_pretrained(output_dir)
    print(f"LoRA model and configuration saved to {output_dir}")

# ----- Clean Up Distributed Group -----
if dist.is_initialized():
    dist.destroy_process_group()

