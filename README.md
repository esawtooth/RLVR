## GSM8K-RLVR: Reinforcement Learning from Verifiable Rewards for Base Models

This repository explores applying the Reinforcement Learning from Verifiable Rewards (RLVR) concept, as proposed in the Tulu3 paper ("Pushing Frontiers in Open Language Model Post-Training") and inspired by DeepSeek R1, to enhance the performance of base language models on the GSM8K math problem-solving dataset.

### Key Features:

- Base Model Focus: Targets improving base models without reliance on pre-trained reward models.

- Few-Shot Prompting: Leverages few-shot examples within each model's input to establish the desired data pattern and facilitate RL learning.

- Simplified Prompt Format: Omits explicit `<think>` and `<answer>` tags, keeping completions straightforward.

- Dual Reward System: Employs two reward functions:

    - Correctness: Rewards accurate answers to GSM8K problems.

    -  Format Adherence: Incentivizes outputting the final answer in the specific `#### The final answer is {number}` format.

### Goal:

To adapt and refine base models for better mathematical reasoning and structured output through RLVR, using only GSM8K data and few-shot prompting.