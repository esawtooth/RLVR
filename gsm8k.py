# gsm8k.py

import logging
import random
from resources import EXAMPLARS
from datasets import load_dataset

logger = logging.getLogger(__name__)

class GSM8K:
    def __init__(self, split, include_answer=True, include_reasoning=True, few_shot=False, num_shots=8, seed=None, cot=False, template="qa"):
        self.split = split
        self.include_answer = include_answer
        self.include_reasoning = include_reasoning
        self.seed = seed

        if self.seed is not None:
            random.seed(self.seed)

        self.few_shot = few_shot 
        self.num_shots = num_shots
        self.cot = cot
        self.template = template
        self.examples = None
        self.dataset = self.load_dataset()
        
    def format_example(self, question, solution, answer):
        example = ''
  
        if self.template == 'qa':
            example = f"Question: {question}\nSolution: "
            
            if self.cot:
                example += "Let's break it down step by step. "
            
            # example += '\n'
            
            if solution is not None:
                def remove_placeholders(text):
                    import re
                    # Regex to match <<anything>>
                    cleaned_text = re.sub(r'<<.*?>>', '', text)
                    return cleaned_text

                solution = '. '.join(solution.split('\n'))
                solution = remove_placeholders(solution)
                example += f"{solution}.\n"
            
            example = example.replace('..', '.')
            
            if answer is not None:
                example += f"#### The final answer is {answer}\n\n"   
        else:
            raise ValueError('Format Not Implemented')
        
        return example  
    
    def process_example(self, example, index):
        # Set seed based on self.seed and index for reproducibility
        if self.seed is not None:
            random.seed(self.seed + index)

        # Always capture the original raw question
        question = example['question']

        # Rest of the original method remains the same
        answer = example['answer']
        
        # Extract the reasoning steps and the final answer
        answer_delim = "#### "
        if answer_delim in answer:
            reasoning = answer.split(answer_delim)[0].strip()
            final_answer = answer.split(answer_delim)[-1].strip()
        else:
            reasoning = answer.strip()
            final_answer = ''
            
        # Create the code solution
        if self.include_answer:
            if self.include_reasoning:
                input_text = self.format_example(question, reasoning, final_answer)
            else:
                input_text = self.format_example(question, None, final_answer)
        else:
            input_text = self.format_example(question, None, None)

        if self.few_shot:
            input_text = self.few_shot_prompt + input_text
            
        return {
            'text': input_text,
            'final_answer': final_answer,
            'question': question, 
        }    

    def load_dataset(self):
        # Load the GSM8K dataset with the specified split
        dataset = load_dataset('gsm8k', 'main', split=self.split)
        
        if self.few_shot:
            self.few_shot_prompt = self.build_prompt()

        # Use with_indices=True to pass the index to the processing function
        dataset = dataset.map(self.process_example, with_indices=True, load_from_cache_file=False)

        return dataset
    
    def fewshot_examples_qa(self):
        """Loads and returns the few-shot examples for the task if they exist."""
        
        return EXAMPLARS

    def make_prompts(self):
        """Builds the prompt for the LM to generate from."""

        if self.template == 'qa':
            examples = self.fewshot_examples_qa()
        else:
            raise ValueError('Format Not Implemented')
    
        self.examples = examples        
    
    def build_prompt(self):
        
        if self.examples is None:
            self.make_prompts()
        
        random.seed(42)
        
        prompt = ""
        for qna in random.sample(self.examples, self.num_shots):
            prompt += self.format_example(qna['question'], qna['cot_answer'], qna['short_answer'])
            
        return prompt
