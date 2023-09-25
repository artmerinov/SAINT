import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


def get_dataset(config):

    data = pd.read_csv(
        filepath_or_buffer=config.DATASOURCE, 
        usecols=[
            'user_id', 
            'content_id',
            'answered_correctly',
            'user_answer',
            'content_type_id',
        ], 
        dtype={
            'user_id':'int32' ,
            'content_id':'int16',
            'answered_correctly':'int8',
            'user_answer': 'int8',
            'content_type_id': 'int8',
        },
        nrows=1e7,
    )

    data = data.loc[data["content_type_id"] == False]
    data = data.drop(columns=['content_type_id'])

    # filter
    vc = data.groupby('user_id')['content_id'].count()
    trusted_users = vc[vc > 5].index
    data = data.loc[data['user_id'].isin(trusted_users)]
    
    data = data.set_index("user_id")
    return data


def train_test_split(data, config):
    tr_data = data[:int(len(data)*config.TR_FRAC)]
    va_data = data[int(len(data)*config.TR_FRAC):]
    return tr_data, va_data


def calc_causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


class RIIIDDataset(Dataset):
    def __init__(self, dataset, config):
        super().__init__()
        self.dataset = dataset
        self.users = self.dataset.index.unique()
        self.config = config

    def __len__(self):
        return len(self.users)
    
    def __getitem__(self, idx):

        user_id = self.users[idx]
        user_df = self.dataset.loc[user_id]
        seq_len = len(user_df)

        user_e = 1 + torch.from_numpy(user_df['content_id'].values)
        user_c = 1 + torch.from_numpy(user_df['answered_correctly'].values)
        user_a = 1 + torch.from_numpy(user_df['user_answer'].values)

        e = torch.zeros(size=(self.config.MAX_LEN,), dtype=torch.int32) # exercise id
        c = torch.zeros(size=(self.config.MAX_LEN,), dtype=torch.int32) # correctness (True, False)
        a = torch.zeros(size=(self.config.MAX_LEN,), dtype=torch.int32) # user answer (A, B, C, D)

        if seq_len < self.config.MAX_LEN:

            # add padding in front
            e[-seq_len:] = user_e
            c[-seq_len:] = user_c
            a[-seq_len:] = user_a

        elif seq_len > self.config.MAX_LEN:

            # uniformly select starting point 
            st = np.random.randint(low=0, high=seq_len - self.config.MAX_LEN)
            en = st + self.config.MAX_LEN

            e[:] = user_e[st:en]
            c[:] = user_c[st:en]
            a[:] = user_a[st:en]

        else:

            e[:] = user_e
            c[:] = user_c
            a[:] = user_a
            

        assert e.size(0) == self.config.MAX_LEN
        assert c.size(0) == self.config.MAX_LEN
        assert a.size(0) == self.config.MAX_LEN

        src_mask = (e[1:] != 0).int().unsqueeze(0).unsqueeze(0)

        tgt_mask = (e[:-1] != 0).int().unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_mask & calc_causal_mask(e.size(0) - 1)

        return {
            # encoder input (current)
            "e": e[1:],

            # decoder input (history)
            'c': c[:-1],
            'a': a[:-1],

            # label
            'label': c[1:],

            # masks
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
        }
