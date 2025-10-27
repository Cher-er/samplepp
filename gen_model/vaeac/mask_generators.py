import numpy as np
import torch
import pandas as pd
import random


# Mask generator for missing feature imputation

class MCARGenerator:
    """
    Returned mask is sampled from component-wise independent Bernoulli
    distribution with probability of component to be unobserved p.
    Such mask induces the type of missingness which is called
    in literature "missing completely at random" (MCAR).

    If some value in batch is missed, it automatically becomes unobserved.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        nan_mask = torch.isnan(batch).float()  # missed values
        bernoulli_mask_numpy = np.random.choice(2, size=batch.shape, p=[1 - self.p, self.p])
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()
        mask = torch.max(bernoulli_mask, nan_mask)  # logical or
        return mask


class StratifiedMask:
    def __init__(self, r, data: pd.DataFrame, schema):
        self.r = r
        self.data = data
        self.schema = schema

        self.unique_counts = {}
        self.value_row_counts = {}
        for col in data.columns:
            self.unique_counts[col] = data[col].nunique()
            
            self.value_row_counts[col] = data[col].value_counts().to_dict()
        
    def __call__(self, batch):
        nan_mask = torch.isnan(batch).float()   

        columns = list(self.schema.keys())
        mask = pd.DataFrame(np.zeros(nan_mask.shape), columns=columns, dtype=int)
        rows_n = int(self.r * batch.shape[0])

        for col_idx, (col, col_type) in enumerate(self.schema.items()):
            if col_type == 'c':
                if col not in self.value_row_counts:
                    raise ValueError(f"Value counts for column '{col}' not provided.")
                col_value_counts = self.value_row_counts[col]
                total_count = sum(col_value_counts.values())
                selected_indices = random.sample(range(batch.shape[0]), rows_n)
                for idx in selected_indices:
                    value = batch[idx][col_idx].item()
                    if pd.isna(value) or value not in col_value_counts:
                        continue
                    probability = col_value_counts[value] / total_count
                    if random.random() < probability:
                        mask.iloc[idx, col_idx] = 1
            elif col_type == 'n':
                mask[col] = 1
            else:
                raise ValueError(f"Unknown column type: {col_type} for column {col}")
        mask = torch.from_numpy(mask.values).float()
        mask = torch.max(mask, nan_mask)
        return mask


class RandomStratifiedMask:
    def __init__(self, r, data: pd.DataFrame, schema):
        self.r = r
        self.data = data
        self.schema = schema

        self.unique_counts = {}
        self.value_row_counts = {}
        for col in data.columns:
            self.unique_counts[col] = data[col].nunique()
            
            self.value_row_counts[col] = data[col].value_counts().to_dict()
        
    def __call__(self, batch):
        nan_mask = torch.isnan(batch).float()   

        columns = list(self.schema.keys())
        mask = pd.DataFrame(np.zeros(nan_mask.shape), columns=columns, dtype=int)
        rows_n = int(self.r * batch.shape[0])

        for col_idx, (col, col_type) in enumerate(self.schema.items()):
            selected_indices = random.sample(range(batch.shape[0]), rows_n)

            if col_type == 'c':
                if col not in self.value_row_counts:
                    raise ValueError(f"Value counts for column '{col}' not provided.")
                col_value_counts = self.value_row_counts[col]
                total_count = sum(col_value_counts.values())
                for idx in selected_indices:
                    value = batch[idx][col_idx].item()
                    if pd.isna(value) or value not in col_value_counts:
                        continue
                    probability = col_value_counts[value] / total_count
                    if random.random() < probability:
                        mask.iloc[idx, col_idx] = 1
            elif col_type == 'n':
                mask.loc[selected_indices, col] = 1
            else:
                raise ValueError(f"Unknown column type: {col_type} for column {col}")
        mask = torch.from_numpy(mask.values).float()
        mask = torch.max(mask, nan_mask)
        return mask