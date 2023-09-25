import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd


def get_dataset(config):

    # load train data
    data = pd.read_csv(
        filepath_or_buffer=config.DATASOURCE, 
        usecols=[
            'user_id', 
            'content_id',
            'answered_correctly',
            'user_answer',
            'content_type_id',
            'timestamp',
        ], 
        dtype={
            'user_id':'int32' ,
            'content_id':'int16',
            'answered_correctly':'int8',
            'user_answer': 'int8',
            'content_type_id': 'int8',
            'timestamp': 'int64',
        },
        nrows=1e7,
    )

    # remove lectures
    data = data.loc[data["content_type_id"] == False]
    data = data.drop(columns=['content_type_id'])

    # add part
    questions = pd.read_csv(
        filepath_or_buffer='data/questions.csv', 
        usecols={
            'question_id',
            'part'
        },
        dtype={
            'question_id': 'int16',
            'part': 'int8',
        }
    )

    question_part_mapping = questions.set_index('question_id')['part'].to_dict()
    data['part'] = data['content_id'].transform(lambda x: question_part_mapping[x])

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

        # initialise vectors with zeros
        ex = torch.zeros(size=(self.config.MAX_LEN,), dtype=torch.int32) # exercise id (0, 1, ..., 13523)
        ac = torch.zeros(size=(self.config.MAX_LEN,), dtype=torch.int32) # answered correctly (0, 1)
        ua = torch.zeros(size=(self.config.MAX_LEN,), dtype=torch.int32) # user answer (0, 1, 2, 3)
        ep = torch.zeros(size=(self.config.MAX_LEN,), dtype=torch.int32) # exercise part (1, 2, 3, 4, 5, 6, 7)

        # add 1 to some ids because we will add zero padding
        # we don't want to mix padding id "0" with potential id "0"
        user_ex = 1 + torch.from_numpy(user_df['content_id'].values) # (0, 1, ..., 13523) --> (1, 2, ..., 13524) --> 
        user_ac = 1 + torch.from_numpy(user_df['answered_correctly'].values) # (0, 1) --> (1, 2)
        user_ua = 1 + torch.from_numpy(user_df['user_answer'].values) # (0, 1, 2, 3) --> (1, 2, 3, 4)
        user_ep = 0 + torch.from_numpy(user_df['part'].values) # no need to shift

        if seq_len < self.config.MAX_LEN:

            # add padding in front of sequence
            ex[-seq_len:] = user_ex
            ac[-seq_len:] = user_ac
            ua[-seq_len:] = user_ua
            ep[-seq_len:] = user_ep

        elif seq_len > self.config.MAX_LEN:

            # uniformly select starting point 
            st = np.random.randint(low=0, high=seq_len - self.config.MAX_LEN)
            en = st + self.config.MAX_LEN

            ex[:] = user_ex[st:en]
            ac[:] = user_ac[st:en]
            ua[:] = user_ua[st:en]
            ep[:] = user_ep[st:en]

        else:

            ex[:] = user_ex
            ac[:] = user_ac
            ua[:] = user_ua
            ep[:] = user_ep
            
        assert ex.size(0) == self.config.MAX_LEN
        assert ac.size(0) == self.config.MAX_LEN
        assert ua.size(0) == self.config.MAX_LEN
        assert ep.size(0) == self.config.MAX_LEN

        src_mask = (ex[1:] != 0).int().unsqueeze(0).unsqueeze(0)
        
        tgt_mask = (ex[:-1] != 0).int().unsqueeze(0).unsqueeze(0)
        tgt_mask = tgt_mask & calc_causal_mask(ex.size(0) - 1)

        return {
            # encoder input (current)
            "ex": ex[1:],
            "ep": ep[1:],

            # decoder input (history)
            'ac': ac[:-1],
            'ua': ua[:-1],

            # label
            'label': ac[1:],

            # masks
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
        }


def calc_causal_mask(size):
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0