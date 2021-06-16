import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from transformers import BertTokenizer, BertModel
from transformers import AutoTokenizer, AutoModel
import torch
from allennlp.modules import ConditionalRandomField
from sentence_transformers import SentenceTransformer, util

import os
import pprint

pp = pprint.PrettyPrinter(indent=4)



class GRADE_model(nn.Module):
    """
    This class represents a full neural Learning to Rank model with a given encoder model.
    """
    def __init__(self, startpath, level):
        """
        :param input_layer: the input block (e.g. FCModel)
        :param encoder: the encoding block (e.g. transformer.Encoder)
        :param output_layer: the output block (e.g. OutputLayer)
        """
        super(GRADE_model, self).__init__()

        self.WINDOWSIZE = 5
        self.NUMCLASSES = level

        # (1) Pretrained bert
        #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')

        # (2) Pretrained conversational bert
        #self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-cased-conversational")
        #self.bert_encoder = AutoModel.from_pretrained("DeepPavlov/bert-base-cased-conversational")

        if os.path.isdir(startpath):
            #self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            #self.bert_encoder = BertModel.from_pretrained('bert-base-uncased')
            # Must use DeepPavolv for initialization to for dimension to be matched
            self.tokenizer = AutoTokenizer.from_pretrained("DeepPavlov/bert-base-cased-conversational")
            self.bert_encoder = AutoModel.from_pretrained("DeepPavlov/bert-base-cased-conversational")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(startpath)
            self.bert_encoder = AutoModel.from_pretrained(startpath)

        #print(self.bert_encoder.max)
        #exit()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        #self.input_norm = nn.LayerNorm(64)
        self.l1 = torch.nn.Linear(768, 256)
        self.relu = torch.nn.ReLU()
        self.l2 = torch.nn.Linear(256, self.NUMCLASSES)

        #self.crf = ConditionalRandomField(self.NUMCLASSES)




    def prep_window(self, window):
        joined = f' {self.tokenizer.sep_token} '.join([' '.join(utt.split()[:int(512.0/(2*self.WINDOWSIZE))]) for utt in window])
        return joined

    def forward(self, windows):
        """
        Forward pass through the input layer and encoder.
        :param x: input of shape [batch_size, slate_length, input_dim]
        :param mask: padding mask of shape [batch_size, slate_length]
        :param indices: original item ranks used in positional encoding, shape [batch_size, slate_length]
        :return: encoder output of shape [batch_size, slate_length, encoder_output_dim]
        """
        # TODO: do bert encoding here, create masks, indices etc
        windows = [ self.prep_window(window) for window in windows ]


        # (1), (2)
        tokenized = self.tokenizer(windows, return_tensors='pt', truncation=True, padding=True, max_length=512)

        sep_mask = tokenized["input_ids"] == self.tokenizer.sep_token_id

        tokenized = tokenized.to(self.device)

        output = self.bert_encoder(**tokenized).last_hidden_state


        sep_embedding = output[sep_mask]

        scores = self.l2(self.relu(self.l1(sep_embedding)))


        return scores




def make_GRADE_model(startpath, level):
    print(f'Using model {startpath}')

    model = GRADE_model(startpath, level)
    if os.path.isdir(startpath):
        model.load_state_dict(torch.load(startpath+"/weights.pth"))
        model.tokenizer.from_pretrained(startpath)
    return model