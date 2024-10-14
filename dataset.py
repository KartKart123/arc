import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from count import count_function_lines

class ARCDataset(Dataset):
    def __init__(self, dataset_dir, split="train"):
        self.dataset_dir = dataset_dir
        self.split = split
        self.task_files = [f for f in os.listdir(dataset_dir) if f.endswith('.json')]
        self.data = self._load_tasks()

    def _load_tasks(self):
        line_counts = count_function_lines('solvers.py')
        data = []
        for task_file in self.task_files:
            task_path = os.path.join(self.dataset_dir, task_file)
            line_count = line_counts[task_file]
            with open(task_path, 'r') as f:
                task = json.load(f)
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


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_grid, output_grid, line_count = self.data[idx]
        return input_grid, output_grid, line_count

""" 
dataset_dir = "data\\training"
arc_train_dataset = ARCDataset(dataset_dir=dataset_dir, split="train")

first_train_sample = arc_train_dataset[0]
input_grid = first_train_sample['input']
print("Input Grid:", input_grid)
print("Line Count:", first_train_sample['line_count']) """