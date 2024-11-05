import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from count import count_function_lines

class ARCDataset(Dataset):
    def __init__(self, dataset_dir, split="train", include_mutations=False):
        self.dataset_dir = dataset_dir
        self.split = split
        self.task_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json') and f != "mutated_tasks_train_9600.json"]
        
        # Load regular tasks
        self.data = self._load_tasks()
        
        # Load mutation tasks if requested
        if include_mutations and split == "train":
            print("Loading mutation tasks")
            mutation_file = "mutated_tasks_train_9600.json"
            if os.path.exists(os.path.join(dataset_dir, mutation_file)):
                mutation_data = self._load_mutation_tasks(mutation_file)
                self.data.extend(mutation_data)

    def _load_tasks(self):
        line_counts = count_function_lines('solvers.py')
        data = []
        for task_file in self.task_files:
            task_path = os.path.join(self.dataset_dir, task_file)
            line_count = line_counts[task_file]
            with open(task_path, 'r') as f:
                task = json.load(f)
                # if isinstance(task, list):
                #     print("Number of examples:", len(task))
                #     print("First example structure:", task[0].keys(), task[0].values())
                task_data = task[self.split] 
                for sample in task_data:
                    input_grid = np.array(sample['input'])
                    input_grid = self.pad_to_30x30(input_grid)
                    output_grid = np.array(sample['output'])
                    output_grid = self.pad_to_30x30(output_grid)
                    data.append((input_grid, output_grid, line_count))  
        return data
    
    def pad_to_30x30(self, grid, pad_value=0):
        padded_grid = torch.full((30, 30), pad_value, dtype=torch.float32)
        height, width = len(grid), len(grid[0])
        padded_grid[:height, :width] = torch.tensor(grid, dtype=torch.float32)/10.0
        return padded_grid

    def _count_program_lines(self, program_str):
        """Count lines in program body, excluding def, return, empty lines, and comments"""
        lines = program_str.strip().split('\n')
        count = 0
        for line in lines:
            line = line.strip()
            if (line and 
                not line.startswith('def') and 
                not line.startswith('return') and 
                not line.startswith('#')):
                count += 1
        return count

    def _load_mutation_tasks(self, mutation_file):
        data = []
        
        with open(os.path.join(self.dataset_dir, mutation_file), 'r') as f:
            mutation_data = json.load(f)
        
        total_tasks = 0
        total_valid_tasks = 0
        # Iterate through each task in the dictionary
        for task_id, task_info in mutation_data.items():
            # Get training examples from the task
            index = "training_examples" if self.split == "train" else "test_examples"
            for example in task_info[index]:
                total_tasks += 1
                input_grid = np.array(example['input'])
                output_grid = np.array(example['output'])

                 # Skip degenerate examples
                if len(input_grid.shape) != 2 or len(output_grid.shape) != 2:
                    continue
               
                # Skip examples that exceed 30x30 dimensions
                if input_grid.shape[0] > 30 or input_grid.shape[1] > 30 or \
                   output_grid.shape[0] > 30 or output_grid.shape[1] > 30:
                    continue

                total_valid_tasks += 1
                # Count actual lines of code using same logic as count_function_lines
                line_count = self._count_program_lines(task_info['program'])
                
                # Use existing padding method
                padded_input = self.pad_to_30x30(input_grid)
                padded_output = self.pad_to_30x30(output_grid)
                
                data.append((padded_input, padded_output, line_count))
            # print(f"Finished processing {task_id}")

        print("Total training tasks: ", total_tasks)
        print("Total valid tasks: ", total_valid_tasks) 
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_grid, output_grid, line_count = self.data[idx]
        return input_grid, output_grid, line_count

""" 
dataset_dir = os.path.join("data", "training") 
arc_train_dataset = ARCDataset(dataset_dir=dataset_dir, split="train")

first_train_sample = arc_train_dataset[0]
input_grid = first_train_sample['input']
print("Input Grid:", input_grid)
print("Line Count:", first_train_sample['line_count'])

# Without mutations (original behavior)
dataset = ARCDataset(dataset_dir, split="train")

# With mutations
dataset = ARCDataset(dataset_dir, split="train", include_mutations=True)

"""