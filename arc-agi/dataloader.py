from datasets import Dataset, DatasetDict
from typing import Any, Dict



def format_grid(grid_input):
    # If input is already a list, use it directly
    if isinstance(grid_input, list):
        grid = grid_input
    else:
        # Convert string representation to list if needed
        import ast
        grid = ast.literal_eval(grid_input)
    
    # Format each row with spaces between numbers
    formatted_rows = []
    for row in grid:
        formatted_row = ' '.join(str(num) for num in row)
        formatted_rows.append(formatted_row)
    
    # Join rows with newlines
    return '\n'.join(formatted_rows)


def format_prompt(example):
    # Start with empty prompt
    prompt = "Find the common rule that maps an input grid to an output grid given the examples below.\n\n"
    
    # Add each training example as context
    for idx, train_example in enumerate(example['train']):
        prompt += f"Example {idx + 1}:\n\n"
        prompt += f"Input:\n{format_grid(train_example['input'])}\n"  # Assuming input is already in list format
        prompt += f"Output:\n{format_grid(train_example['output'])}\n\n"
    
    # Add the test question
    prompt += "\nBelow is a test input grid. Predict the corresponding output grid by applying the rule you found. Keep in mind that your thinking maybe abrubtly terminated with '[THINKING TIME UP]' and so you must answer only with the thinking tokens you have thus far.\n\n"
    prompt += f"Input:\n{format_grid(example['test'][0]['input'])}\n"
    prompt += "Respond with the corresponding output. You must only respond with the output or you will be penalized for extra tokens. Reinforce the output format via your thinking."
    
    return prompt


class SelfAdaptingDataset:
    def __init__(self, dataset):
        self.original_dataset = dataset
        self.formatted_dataset = self._transform_dataset()
    
    def _transform_single_example(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Transform a single example from train/test format to prompt/completion format."""
        prompt = f'<|user|>{format_prompt(example)}<|user|><|assistant|><think>\n'
        completion = example['test'][0]['output']
        
        return {
            'prompt': prompt,
            'completion': completion,
            'learned': '',  # Empty placeholder for learned field
            'mistake': '',  # Empty placeholder for mistake field
            'best_completion': '',  # Empty placeholder for best completion field
            'original_data': example  # Keep original data for reference if needed
        }
    
    def _transform_dataset(self) -> DatasetDict:
        """Transform the entire dataset."""
        
        def transform_split(split_dataset):
            transformed_data = {
                'prompt': [],
                'completion': [],
                'learned': [],
                'mistake': [],
                'best_completion': [],
                'original_data': []
            }
            
            for example in split_dataset:
                transformed = self._transform_single_example(example)
                for key in transformed_data:
                    transformed_data[key].append(transformed[key])
            
            return Dataset.from_dict(transformed_data)
        
        # Transform each split
        transformed_dataset = DatasetDict({
            'training': transform_split(self.original_dataset['training']),
            'evaluation': transform_split(self.original_dataset['evaluation'])
        })
        
        return transformed_dataset
    
    def get_dataset(self) -> DatasetDict:
        """Get the transformed dataset."""
        return self.formatted_dataset
    
    def update_example(self, index: int, split: str, 
                      learned: str = None, 
                      mistake: str = None, 
                      best_completion: str = None):
        """
        Update the learned/mistake/best_completion fields for a specific example.
        
        Args:
            index: Index of the example to update
            split: 'training' or 'evaluation'
            learned: What the model learned from this example
            mistake: What mistake was made
            best_completion: The best completion found so far
        """
        if learned is not None:
            self.formatted_dataset[split] = self.formatted_dataset[split].map(
                lambda x, i: {'learned': learned} if i == index else {'learned': x['learned']},
                with_indices=True
            )
        
        if mistake is not None:
            self.formatted_dataset[split] = self.formatted_dataset[split].map(
                lambda x, i: {'mistake': mistake} if i == index else {'mistake': x['mistake']},
                with_indices=True
            )
        
        if best_completion is not None:
            self.formatted_dataset[split] = self.formatted_dataset[split].map(
                lambda x, i: {'best_completion': best_completion} if i == index else {'best_completion': x['best_completion']},
                with_indices=True
            )