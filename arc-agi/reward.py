import re
import numpy as np
from typing import Optional, Union, List
from vllm.sampling_params import SamplingParams
import json
from datetime import datetime
from pathlib import Path
from filelock import FileLock



class ThinkingCompletionRewardFunction:
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.log_file = self.log_dir / f"rewards_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        self.file_lock = FileLock(str(self.log_file) + ".lock")
        

    def log_entry(self, prompt: str, thoughts: str, answer: str, expected: str, reward: float):
        """
        Thread-safe logging of reward computation entries to a JSONL file.
        
        Args:
            prompt: The input prompt
            thoughts: The model's thinking process
            answer: The model's answer
            expected: The expected completion
            reward: The computed reward
        """
        entry = {
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "thoughts": thoughts,
            "answer": answer,
            "expected": expected,
            "reward": reward
        }

        # Use file lock to ensure thread-safe writing
        with self.file_lock:
            with open(self.log_file, "a") as f:
                json.dump(entry, f)
                f.write("\n")

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
            return -0.25
        
        # Get the last grid and apply penalty for multiple grids
        gen_array = all_grids[-1]
        reward = 0.0 if len(all_grids) == 1 else -0.1
        
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

    def calculate_reward(self, prompts: List[str], completions: List[str], **kwargs) -> List[float]:
        # Generate final completions using the thinking
        
        answers = kwargs.get("answers", [])
        expected_completions = kwargs.get("expected_completions", [])[0]
        # Compute rewards
        rewards = []
        print("Lengths: ", len(prompts), len(completions), len(answers), len(expected_completions))
        assert(len(prompts) == len(completions) == len(answers) == len(expected_completions))

        for prompt, thoughts, answer, expected in zip(prompts, completions, answers, expected_completions):
            reward = self._compute_reward(answer, expected)
            rewards.append(reward)

            # Log the entry
            self.log_entry(
                prompt=prompt,
                thoughts=thoughts,
                answer=answer,
                expected=expected,
                reward=reward
            )
            
        print("Final reward vector: ", rewards)
        return rewards