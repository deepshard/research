import re
import numpy as np
from typing import Optional, Union, List
from vllm.sampling_params import SamplingParams

class ThinkingCompletionRewardFunction:
    def __init__(self, model, tokenizer, max_completion_tokens: int):

    def parse_grid(self, text: str) -> list[list[int]]:
        """
        Parses a 2D grid of integers from a given text string.
        Any line that can be split purely into integers is assumed to be part of the grid.
        Lines that contain text or cannot be parsed as integers are ignored.
        
        :param text: A string containing arbitrary text and lines with integers.
        :return: A list of lists of integers representing the parsed grid.
        """
        main_grid = []
        grid = []
        
        for line in text.splitlines():
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Attempt to parse the entire line as a row of integers
            tokens = line.split()
            try:
                row = [int(token) for token in tokens]
                grid.append(row)
            except ValueError:
                if grid != []:
                    main_grid.append(grid)
                    grid = []
                continue
        
        return main_grid

    def _compute_reward(self, generated: str, expected: str) -> float:
        """
        Compute reward based on the following criteria:
        - Multiple grids detected: -0.25 penalty
        - Dimension match: +0.5 points
        - Full match: 2.0 points total
        - Partial match: Scaled between 0 and 2.0 based on correctness
        """
        # Extract arrays from both strings
        all_grids = self.parse_grid(generated)
        exp_array = expected
        
        # If no grids found or expected array is None, return 0
        if not all_grids or exp_array is None:
            return 0.0
        
        # Get the last grid and apply penalty for multiple grids
        gen_array = all_grids[-1]
        reward = 0.0 if len(all_grids) == 1 else -0.25
        
        try:
            gen_np = np.array(gen_array)
            exp_np = np.array(exp_array)
            
            # Check dimensions
            if gen_np.shape == exp_np.shape:
                reward += 0.5
                
                # Calculate element-wise match percentage
                total_elements = np.prod(gen_np.shape)
                matching_elements = np.sum(gen_np == exp_np)
                match_percentage = matching_elements / total_elements
                
                if match_percentage == 1.0:
                    # Perfect match (including the 0.5 from dimension match)
                    reward = 2.0
                else:
                    # Scale remaining 1.5 points based on match percentage
                    reward += 1.5 * match_percentage
                    
        except:
            return reward  # Return whatever reward we've accumulated so far
        
        return reward

    def __call__(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Generate final completions using the thinking
        answers = kwargs.get("answers", [])
        expected_completions = kwargs.get("completion", [])
        
        # Compute rewards
        rewards = []
        for gen_comp, exp_comp in zip(answers, expected_completions):
            reward = self._compute_reward(gen_comp, exp_comp)
            rewards.append(reward)
            
        return rewards