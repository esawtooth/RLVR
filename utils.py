import re

def format_reward_func(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    # Adjusted pattern to allow matching anywhere in the string
    pattern = r"\n#### The final answer is \d+$"    
    completion_contents = [completion[0] for completion in completions]    
    matches = [re.search(pattern, content) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def reward_func(completions,  final_answers, **kwargs):
    completion_contents = [completion[0] for completion in completions]    
    rewards = []
    
    for completion, final_answer in zip(completion_contents, final_answers) :
        try:
            match = re.search(r'####.*?([\d,]+(?:\.\d+)?)', completion)
            if match:
                answer = match.group(1)
                
                for remove_char in [',', '$', '%', 'g']:
                    answer = answer.replace(remove_char, '')
                    
                if abs(float(answer)-float(final_answer)) < 1e-3:
                    rewards.append(1.0)
                else:
                    rewards.append(0.0)
                
        except ValueError:
            rewards.append(0.0)
            
    return rewards