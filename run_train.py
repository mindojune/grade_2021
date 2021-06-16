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
from torch.utils.data import DataLoader, RandomSampler
import os
from tqdm import tqdm, trange
from torch.nn.utils.rnn import pad_sequence
#
import wandb

os.environ["CUDA_VISIBLE_DEVICES"] = "3"  # 0,1,2,3 check https://docs.google.com/spreadsheets/d/1uLBn3PWVunn8HWkUHZo3hkHqjIDod5vycxEW5_9Ve9w/edit#gid=0
BATCH_SIZE = 8 #16 #32 # 32 is too large
CHUNKSIZE = 5  # 3#2
# Parameters:
lr = 2e-5  # 0.001 #5e-5 #0.001 #2e-5#0
adam_epsilon = 1e-8
EPOCHS = 1 #5 #2 #1 #2
epochs = EPOCHS
FULL = True #False #True





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

def train(datapath, startpath, level):

    if level==8:
        code2idx = {'D': 0, 'M': 1, 'S': 2, 'H': 3, 'F': 4, 'O': 5, 'E': 6, 'NA': 7}
    elif level==5:
        code2idx = {'D': 0, 'M': 1, 'S': 2, 'H': 2, 'F': 3, 'O': 3, 'E': 0, 'NA': 4}
    elif level==2:
        code2idx = {'D': 0, 'M': 0, 'S': 0, 'H': 0, 'F': 0, 'O': 0, 'E': 0, 'NA': 1}
    idx2code = {code2idx[code]: code for code in code2idx}

    print("Using GPU devices:", os.environ["CUDA_VISIBLE_DEVICES"])

    with open(datapath, "rb") as fh:
        data_dict = pickle.load(fh)

    dataset = []
    for data_id in data_dict:
        window = data_dict[data_id]
        dataset.append(window)

    if FULL:
        n = len(dataset)
    else:
        n = 80
    dataset = dataset[:n]
    print("Dataset Length in # Windows:", len(dataset))

    wandb.init(project='GRADE', entity='dojmin')


    res = []

    kf = KFold(n_splits=5, shuffle=True)

    for train_index, test_index in kf.split(dataset):
        # Make model here
        text_model = make_GRADE_model(startpath, level)
        wandb.watch(text_model)

        # create DataLoaders HERE
        X = np.array([[y for y in x] for x in dataset], dtype=object)
        Y = np.array([[ code2idx[y["code"]] for y in x] for x in dataset], dtype=object)

        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]

        train_set = GRADEDataset(X_train, Y_train)
        test_set = GRADEDataset(X_test, Y_test)

        train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, sampler=RandomSampler(train_set), \
                                  drop_last=True, collate_fn=collate_func)
        val_loader = DataLoader(dataset=test_set, batch_size=BATCH_SIZE, sampler=RandomSampler(test_set), \
                                drop_last=True, collate_fn=collate_func)

        # text_model.train()
        criterion = torch.nn.CrossEntropyLoss()

        num_warmup_steps = 0
        num_training_steps = len(train_loader) * epochs

        ### In Transformers, optimizer and schedules are splitted and instantiated like this:
        optimizer = AdamW(text_model.parameters(), lr=lr, eps=adam_epsilon)  # ,
        # correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps,
                                                    num_training_steps=num_training_steps)  # PyTorch scheduler


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # device = 'cpu'
        print("Device:", device)

        ## Store our loss and accuracy for plotting
        train_loss_set = []
        learning_rate = []

        text_model.to(device)
        text_model.zero_grad()

        losses = []
        # tnrange is a tqdm wrapper around the normal python range
        for _ in trange(1, epochs + 1, desc='Epoch'):
            print("<" + "=" * 22 + F" Epoch {_} " + "=" * 22 + ">")
            # Calculate total loss for this epoch
            batch_loss = 0

            # TODO: let's use sentence embedder model to encode (non-differentiably)

            for step, batch in enumerate(tqdm(train_loader)):
                text_model.train()
                utts = [ [ dic2sentence(utt) for utt in window[0]  ] for window in batch ]
                utts = np.array(utts, dtype=object)#[:, 0]

                # I'm sending labels to device here since loss is to be computed here
                # Just out of convenience... might be confusing later
                labels = np.array(batch, dtype=object)[:, 1]
                labels = [torch.LongTensor(list(label)) for label in labels]
                labels = pad_sequence(labels, batch_first=True, padding_value=-1)
                labels = labels.to(device)

                scores = text_model.forward(utts)

                label_mask = labels != -1
                labels = labels[label_mask]

                loss = criterion(scores, labels)

                #print("loss:", loss.item())
                # Backward pass
                loss.backward()
                losses.append(loss.item())
                # monitoring
                wandb.log({"loss": loss.item()})

                # Clip the norm of the gradients to 1.0
                # Gradient clipping is not in AdamW anymore
                torch.nn.utils.clip_grad_norm_(text_model.parameters(), 1.0)

                # Update parameters and take a step using the computed gradient
                optimizer.step()

                # Update learning rate schedule
                scheduler.step()

                # Clear the previous accumulated gradients
                optimizer.zero_grad()

                # Update tracking variables
                batch_loss += loss.item()

            # Calculate the average loss over the training data.
            avg_train_loss = batch_loss / len(train_loader)


            # store the current learning rate
            for param_group in optimizer.param_groups:
                print("\n\tCurrent Learning rate: ", param_group['lr'])
                learning_rate.append(param_group['lr'])

            train_loss_set.append(avg_train_loss)
            print(F'\n\tAverage Training loss: {avg_train_loss}')

        # Validation
        # Put model in evaluation mode to evaluate loss on the validation set
        text_model.eval()

        # Tracking variables
        #eval_acc = 0.0
        nb_eval_steps = 0.0

        # Evaluate data for one epoch
        predictions, targets = [], []
        for step, batch in enumerate(tqdm(val_loader)):
            utts = [[dic2sentence(utt) for utt in window[0]] for window in batch]
            utts = np.array(utts, dtype=object)  # [:, 0]

            # I'm sending labels to device here since loss is to be computed here
            # Just out of convenience... might be confusing later
            labels = np.array(batch, dtype=object)[:, 1]
            labels = [torch.LongTensor(list(label)) for label in labels]
            labels = pad_sequence(labels, batch_first=True, padding_value=-1)
            labels = labels.to(device)

            with torch.no_grad():
                scores = text_model.forward(utts)

            label_mask = labels != -1
            labels = labels[label_mask]

            # preds = preds.to("cpu")
            scores = scores.to("cpu")
            labels = labels.to("cpu")

            preds = torch.argmax(scores,dim=-1)

            labels = labels.to("cpu")

            predictions+=torch.flatten(preds).tolist()
            targets+= torch.flatten(labels).tolist()

            nb_eval_steps += 1


        print()
        acc =  accuracy_score(predictions, targets)
        f1 = f1_score(predictions, targets, labels = list(set(code2idx.values())), average=None, zero_division=0)
        pp.pprint(F'\n\tEpoch {_} Validation Accuracy: {acc}')
        pp.pprint(F'\n\tValidation F1: {f1}')
        res.append([acc, f1])

        # break here to do only 1 cv
        if False:
            break

    model_directory = f'./GRADE{EPOCHS}/'
    if not os.path.isdir(model_directory):
        os.mkdir(model_directory)
    torch.save(text_model.state_dict(),model_directory+'weights.pth')
    text_model.tokenizer.save_pretrained(model_directory)
    pp.pprint(res)

    from operator import add
    temp_f1 = [ 0.0 for x in range(len(res[0][1]))]
    acc = 0
    for result in res:
        acc += result[0]
        temp_f1 = list ( map(add, temp_f1, result[1]))

    acc /= len(res)
    temp_f1 = [ x/len(res) for x in temp_f1 ]

    print("average acc:", acc)
    print("average f1:", temp_f1)
    return text_model

#train()