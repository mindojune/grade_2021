import json
import pickle
import glob
from sklearn.metrics import f1_score
import pprint

pp = pprint.PrettyPrinter(indent=4)

from numpy.random import randint

# Dataset
import torch, sys, os
import numpy as np
import torch.utils.data.dataset
import pandas as pd
from functools import partial
import torch.nn.functional as F
from transformers import BertTokenizer
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import os
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
#
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 0,1,2,3 check https://docs.google.com/spreadsheets/d/1uLBn3PWVunn8HWkUHZo3hkHqjIDod5vycxEW5_9Ve9w/edit#gid=0
BATCH_SIZE = 16 #16 #32 # 32 is too large
CHUNKSIZE = 5  # 3#2
# Parameters:
lr = 2e-5  # 0.001 #5e-5 #0.001 #2e-5#0
adam_epsilon = 1e-8
EPOCHS = 1 #5 #2 #1 #2
epochs = EPOCHS
FULL = True #False #True #False #False #True
n = 80  # 80#000




class GRADEDataset(torch.utils.data.dataset.Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        assert (len(self.X) == len(self.Y))
        return len(self.X)


def collate_func(input):
    return input


# BERT Framework
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from prepare_data import dic2sentence
from model import make_GRADE_model
from functools import reduce

def test(datapath, startpath, level, model=None):
    print("Using GPU devices:", os.environ["CUDA_VISIBLE_DEVICES"])

    with open(datapath, "rb") as fh:
        data_dict = pickle.load(fh)

    dataset = []
    for data_id in data_dict:
        window = data_dict[data_id]
        dataset.append(window)

    if FULL:
        n = len(dataset)

    dataset = dataset[:n]
    print("Dataset Length in # Windows:", len(dataset))

    #wandb.init(project='GRADE', entity='dojmin')



    if model != None:
        text_model = make_GRADE_model(startpath, level)
    else:
        text_model = model
    #wandb.watch(text_model)

    # create DataLoaders HERE
    X = np.array([[y for y in x] for x in dataset], dtype=object)
    Y = np.array([[ None for y in x] for x in dataset], dtype=object)

    data_set = GRADEDataset(X, Y)


    data_loader = DataLoader(dataset=data_set, batch_size=BATCH_SIZE, sampler=SequentialSampler(data_set), \
                            drop_last=True, collate_fn=collate_func)


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = 'cpu'
    print("Device:", device)


    text_model.to(device)
    text_model.zero_grad()

    # Validation
    # Put model in evaluation mode to evaluate loss on the validation set
    text_model.eval()

    result = {}
    # Evaluate data for one epoch
    for step, batch in enumerate(tqdm(data_loader)):

        utts = [[dic2sentence(utt) for utt in window[0]] for window in batch]
        utts = np.array(utts, dtype=object)

        with torch.no_grad():
            scores = text_model.forward(utts)

        scores = scores.to("cpu")
        preds = torch.argmax(scores,dim=-1)

        utts = [[ utt for utt in window[0]] for window in batch]
        #utts = np.array(utts).flatten()
        utts = reduce(lambda x, y: x + y, utts)

        #print (len(utts) , len(preds))

        try:
            assert(len(utts) == len(preds))
        except:
            print(len(utts), len(preds))
            #pp.pprint(utts)
            #pp.pprint(preds)


        for utt, pred in zip(utts, preds):
            sessID = utt["sessionID"]
            file = utt["file"]
            try:
                batch_name = file.split("/")[-2]
                sess_name = file.split("/")[-1]
                sess_key = f'{batch_name}_{sess_name}'
            except:
                sess_key = sessID
            #print(batch_name, sess_name)
            #exit()
            if sess_key not in result:
                result[sess_key] = []
            result[sess_key].append([utt, pred.item()])

        # Temporary rule for testing
        #if step > 100:
        #   break
    return result

