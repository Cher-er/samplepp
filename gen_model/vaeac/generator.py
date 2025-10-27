from gen_model.vaeac.imputation_networks import get_imputation_networks
from gen_model.vaeac.VAEAC import VAEAC
from gen_model.vaeac.train_utils import extend_batch, get_validation_iwae
from gen_model.vaeac.mask_generators import MCARGenerator, StratifiedMask, RandomStratifiedMask

from copy import deepcopy
from math import ceil
from sys import stderr
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import gc

class Generator:
    def __init__(self, raw_data: pd.DataFrame, schema: dict, num_workers=0, load_state=None):
        self.schema = schema
        self.columns = list(schema.keys())
        self.raw_data = raw_data.dropna()
        self.ncol_min = {}
        self.ncol_max = {}
        for col, col_type in self.schema.items():
            if col_type == 'n':
                self.ncol_min[col] = self.raw_data[col].min()
                self.ncol_max[col] = self.raw_data[col].max()
        self.encoded_data, self.mappings = self._encode_categorical_columns(self.raw_data, self.schema)
        self.encoded_data = torch.tensor(self.encoded_data.values, dtype=torch.float32)
        self.one_hot_max_sizes = self._get_one_hot_max_sizes(self.raw_data, self.schema)
        self.norm_mean, self.norm_std = self._compute_normalization(self.encoded_data, self.one_hot_max_sizes)
        self.data = (self.encoded_data - self.norm_mean[None]) / self.norm_std[None]
        self.use_cuda = torch.cuda.is_available()
        self.num_workers = num_workers
        self.networks = get_imputation_networks(self.one_hot_max_sizes)
        self.vlb_scale_factor = self.networks.get('vlb_scale_factor', 1)
        self.epoch = 0
        self.model = VAEAC(
            self.networks['reconstruction_log_prob'],
            self.networks['proposal_network'],
            self.networks['prior_network'],
            self.networks['generative_network']
        )
        self.optimizer = self.networks['optimizer'](self.model.parameters())
        self.validation_iwae = []
        self.train_vlb = []
        if self.use_cuda:
            self.model = self.model.cuda()
        if load_state:
            with open(load_state, 'rb') as f:
                self.best_state = pickle.load(f)
                self.epoch = self.best_state['epoch']
                self.model.load_state_dict(self.best_state['model_state_dict'])
                self.optimizer.load_state_dict(self.best_state['optimizer_state_dict'])
                self.validation_iwae = self.best_state['validation_iwae']
                self.train_vlb = self.best_state['train_vlb']
        else:
            self.best_state = None
        self.batch_size = self.networks['batch_size']

    def train(self, epochs, mask, save_file='best_state.pkl', validations_per_epoch=1, validation_iwae_num_samples=25, validation_ratio=0.15, verbose=True):
        if mask == "ran 0.2":
            mask_generator = MCARGenerator(0.2)
        elif mask == "ran 0.5":
            mask_generator = MCARGenerator(0.5)
        elif mask == "stra 0.2":
            mask_generator = StratifiedMask(0.2, self.raw_data, self.schema)
        elif mask == "stra 0.5":
            mask_generator = StratifiedMask(0.5, self.raw_data, self.schema)
        elif mask == "ranstra 0.2":
            mask_generator = RandomStratifiedMask(0.2, self.raw_data, self.schema)
        elif mask == "ranstra 0.5":
            mask_generator = RandomStratifiedMask(0.5, self.raw_data, self.schema)
        else:
            raise ValueError(f"Unknown mask: {mask}")
        
        data = self.data
        batch_size = self.batch_size
        num_workers = self.num_workers

        # train-validation split
        val_size = ceil(len(data) * validation_ratio)
        val_indices = np.random.choice(len(data), val_size, False)
        val_indices_set = set(val_indices)
        train_indices = [i for i in range(len(data)) if i not in val_indices_set]
        train_data = data[train_indices]
        val_data = data[val_indices]

        # initialize dataloaders
        dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                                num_workers=num_workers, drop_last=False)
        val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=True,
                                    num_workers=num_workers, drop_last=False)
        
        # number of batches after which it is time to do validation
        validation_batches = ceil(len(dataloader) / validations_per_epoch)

        best_state = self.best_state

        use_cuda = self.use_cuda
        vlb_scale_factor = self.vlb_scale_factor

        start_epoch = self.epoch
        model = self.model
        optimizer = self.optimizer
        validation_iwae = self.validation_iwae
        train_vlb = self.train_vlb

        model.train()
        for epoch in range(start_epoch, epochs):
            iterator = dataloader
            avg_vlb = 0
            if verbose:
                print('Epoch %d...' % (epoch + 1), file=stderr, flush=True)
                iterator = tqdm(iterator)
        
            # one epoch
            for i, batch in enumerate(iterator):
        
                # the time to do a checkpoint is at start and end of the training
                # and after processing validation_batches batches
                if any([
                            i == 0 and epoch == 0,
                            i % validation_batches == validation_batches - 1,
                            i + 1 == len(dataloader)
                        ]):
                    val_iwae = get_validation_iwae(val_dataloader, mask_generator, batch_size, model, validation_iwae_num_samples, verbose)
                    validation_iwae.append(val_iwae)
                    train_vlb.append(avg_vlb)
        
                    # if current model validation IWAE is the best validation IWAE
                    # over the history of training, the current state
                    # is saved to best_state variable
                    if max(validation_iwae) <= val_iwae:
                        best_state = deepcopy({
                            'epoch': epoch + 1,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'validation_iwae': validation_iwae,
                            'train_vlb': train_vlb,
                        })

                        with open(save_file, 'wb') as f:
                            pickle.dump(best_state, f)
        
                # if batch size is less than batch_size, extend it with objects
                # from the beginning of the dataset
                batch = extend_batch(batch, dataloader, batch_size)
        
                # generate mask and do an optimizer step over the mask and the batch
                mask = mask_generator(batch)
                optimizer.zero_grad()
                if use_cuda:
                    batch = batch.cuda()
                    mask = mask.cuda()
                vlb = model.batch_vlb(batch, mask).mean()
                (-vlb / vlb_scale_factor).backward()
                optimizer.step()
        
                # update running variational lower bound average
                avg_vlb += (float(vlb) - avg_vlb) / (i + 1)
                if verbose:
                    iterator.set_description('Train VLB: %g' % avg_vlb)
        
        del dataloader
        del val_dataloader
        del train_data
        del val_data
        del self.raw_data
        gc.collect()
        if use_cuda:
            torch.cuda.empty_cache()


    def impute(self, data: pd.DataFrame, num_imputations=10, verbose=True, xt=None):
        batch_size = self.batch_size
        num_workers = self.num_workers
        model = self.model
        use_cuda = self.use_cuda
        networks = self.networks

        encoded_data = data.copy()

        for col, col_type in self.schema.items():
            if col_type == "c":
                if col in self.mappings:
                    cat_to_int = self.mappings[col]["cat_to_int"]
                    encoded_data[col] = encoded_data[col].map(cat_to_int).astype("float")  # 转为 float 保留 NaN
                else:
                    raise ValueError(f"Missing mapping for categorical column: {col}")
            elif col_type == "n":
                continue
            else:
                raise ValueError(f"Unknown column type: {col_type} for column {col}")
        
        encoded_data = torch.tensor(encoded_data.values, dtype=torch.float32)
        impute_data = (encoded_data - self.norm_mean[None]) / self.norm_std[None]

        dataloader = DataLoader(impute_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)

        # prepare the store for the imputations
        results = []
        for i in range(num_imputations):
            results.append([])
        
        iterator = dataloader
        if verbose:
            iterator = tqdm(iterator)
        
        model.eval()
        # impute missing values for all input data
        for batch in iterator:
        
            # if batch size is less than batch_size, extend it with objects
            # from the beginning of the dataset
            batch_extended = batch.clone().detach()
            batch_extended = extend_batch(batch_extended, dataloader, batch_size)
        
            if use_cuda:
                batch = batch.cuda()
                batch_extended = batch_extended.cuda()
        
            # compute the imputation mask
            mask_extended = torch.isnan(batch_extended).float()
        
            # compute imputation distributions parameters
            with torch.no_grad():
                if xt is not None:
                    xt, _ = self._encode_categorical_columns(xt, self.schema, self.mappings)
                    xt = torch.tensor(xt.values, dtype=torch.float32)
                    xt = (xt - self.norm_mean[None]) / self.norm_std[None]
                    xt = self.networks['prior_network'][0](xt)

                samples_params = model.generate_samples_params(batch_extended, mask_extended, num_imputations, xt=xt, use_cuda=use_cuda)
                samples_params = samples_params[:batch.shape[0]]
        
            # make a copy of batch with zeroed missing values
            mask = torch.isnan(batch)
            batch_zeroed_nans = batch.clone().detach()
            batch_zeroed_nans[mask] = 0
        
            # impute samples from the generative distributions into the data
            # and save it to the results
            for i in range(num_imputations):
                sample_params = samples_params[:, i]
                sample = networks['sampler'](sample_params)
                sample[(~mask).bool()] = 0
                sample += batch_zeroed_nans
                results[i].append(sample.clone().detach().to('cpu'))
        
        # concatenate all batches into one [n x K x D] tensor,
        # where n in the number of objects, K is the number of imputations
        # and D is the dimensionality of one object
        for i in range(len(results)):
            results[i] = torch.cat(results[i]).unsqueeze(1)
        result = torch.cat(results, 1)
        
        # reshape result, undo normalization
        result = result.view(result.shape[0] * result.shape[1], result.shape[2])
        result = result * self.norm_std[None] + self.norm_mean[None]

        result = result.numpy()
        result = pd.DataFrame(result, columns=self.columns)

        for col, col_type in self.schema.items():
            if col_type == "c" and col in self.mappings:
                int_to_cat = self.mappings[col]["int_to_cat"]
                result[col] = result[col].round().astype("Int64").map(int_to_cat)
            elif col_type == "n":
                continue
            else:
                raise ValueError(f"Unknown column type: {col_type} for column {col}")
        
        return result
    
    def gen(self, predicates, num_imputations=10, verbose=True, xt=None):
        data = {col: [] for col in self.columns}
        
        for _ in range(num_imputations):
            row = {}
            for col in self.columns:
                predicate_match = [p for p in predicates if p[0] == col]

                for predicate in predicate_match:
                    op = predicate[1]
                    value = predicate[2]

                    if op == "between":
                        if isinstance(value, tuple) and len(value) == 2:
                            row[col] = random.uniform(value[0], value[1])
                        else:
                            raise ValueError(f"Invalid 'between' range for column '{col}': {value}")
                    else:
                        if self.schema[col] == "n":
                            if op == "=":
                                row[col] = value
                            elif op == ">":
                                row[col] = random.uniform(value, self.ncol_max[col])
                            elif op == "<":
                                row[col] = random.uniform(self.ncol_min[col], value)
                            elif op == ">=":
                                row[col] = random.uniform(value, self.ncol_max[col])
                            elif op == "<=":
                                row[col] = random.uniform(self.ncol_min[col], value)
                            else:
                                raise ValueError(f"Unknown operator '{op}' for numerical column '{col}'")
                        elif self.schema[col] == "c":
                            row[col] = value
                        else:
                            raise ValueError(f"Unknown column type: {self.schema[col]} for column {col}")
                if len(predicate_match) == 0:
                    row[col] = np.nan

            for col in self.columns:
                data[col].append(row[col])
            
        data = pd.DataFrame(data)

        return self.impute(data, 1, verbose, xt=xt)

    def _compute_normalization(self, data, one_hot_max_sizes):
        """
        Compute the normalization parameters (i. e. mean to subtract and std
        to divide by) for each feature of the dataset.
        For categorical features mean is zero and std is one.
        i-th feature is denoted to be categorical if one_hot_max_sizes[i] >= 2.
        Returns two vectors: means and stds.
        """
        norm_vector_mean = torch.zeros(len(one_hot_max_sizes))
        norm_vector_std = torch.ones(len(one_hot_max_sizes))
        for i, size in enumerate(one_hot_max_sizes):
            if size >= 2:
                continue
            v = data[:, i]
            v = v[~torch.isnan(v)]
            vmin, vmax = v.min(), v.max()
            vmean = v.mean()
            vstd = v.std()
            norm_vector_mean[i] = vmean
            norm_vector_std[i] = vstd
        return norm_vector_mean, norm_vector_std
    
    def _get_one_hot_max_sizes(self, df, schema):
        one_hot_max_sizes = []
        for col, col_type in schema.items():
            if col_type == "c":
                unique_count = df[col].nunique()
                one_hot_max_sizes.append(unique_count)
            elif col_type == "n":
                one_hot_max_sizes.append(1)
            else:
                raise ValueError(f"Unknown column type: {col_type} for column {col}")
        return one_hot_max_sizes
    
    def _encode_categorical_columns(self, df, schema, mappings=None):
        encoded_df = df.copy()

        if mappings is None:

            mappings = {}
            
            for col, col_type in schema.items():
                if col_type == "c":
                    unique_vals = encoded_df[col].unique()
                    
                    cat_to_int = {cat: idx for idx, cat in enumerate(unique_vals)}
                    int_to_cat = {idx: cat for cat, idx in cat_to_int.items()}
                    
                    encoded_df[col] = encoded_df[col].map(cat_to_int)
                    
                    mappings[col] = {"int_to_cat": int_to_cat, "cat_to_int": cat_to_int}
                elif col_type == "n":
                    continue
                else:
                    raise ValueError(f"Unknown column type: {col_type} for column {col}")
        
        else:
            for col, col_type in schema.items():
                if col_type == "c":
                    encoded_df[col] = encoded_df[col].map(mappings[col]["cat_to_int"])
                elif col_type == "n":
                    continue
                else:
                    raise ValueError(f"Unknown column type: {col_type} for column {col}")
        
        return encoded_df, mappings


# if __name__ == '__main__':
#     import os
#     os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
#     data_file = "/home/yc/dataset/fflights/fflights_100m.csv"
#     schema = {
#         "Quarter": 'c',
#         "Month": 'c',
#         "DayofMonth": 'c',
#         "DayOfWeek": 'c',
#         "Reporting_Airline": 'c',
#         "Origin": 'c',
#         "OriginStateName": 'c',
#         "Dest": 'c',
#         "DestStateName": 'c',
#         "DepDelay": 'n',
#         "TaxiOut": 'n',
#         "ArrDelay": 'n',
#         "TaxiIn": 'n',
#         "AirTime": 'n',
#         "Distance": 'n'
#     }

#     generator = Generator(data_file, delimiter=',', schema=schema)
#     # generator = Generator(data_file, delimiter=',', schema=schema, load_state="/home/yc/project/SamplePP/vaeac/best_states/fflights_ep10_ranstra0.2.pkl")

#     generator.train(10, mask="ran 0.5", save_file="/home/yc/project/SamplePP/vaeac/best_states/fflights_100m_ep10_ran0.5.pkl")
