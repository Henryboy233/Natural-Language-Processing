from torch.utils.data import Dataset 
from collections import Counter 
import string
import pandas as pd
import numpy as np
import torch

from collections import Counter
import string
import torchtext
from torch import nn


class Vocabulary(object):
#"""Class to process text and extract Vocabulary for mapping"""
    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        if token_to_idx is None: 
            token_to_idx = {}
        self._token_to_idx = token_to_idx 
        self._idx_to_token = {idx: token for token, idx in self._token_to_idx.items()}
        self._add_unk = add_unk 
        self._unk_token = unk_token 
        self.unk_index = 1
        if add_unk:
            self.unk_index = self.add_token(unk_token) 

    def to_serializable(self):
    #""" returns a dictionary that can be serialized """
        return {'token_to_idx': self._token_to_idx, 'add_unk': self._add_unk,'unk_token': self._unk_token} 
        
    @classmethod
    def from_serializable(cls, contents):
        return cls(**contents)

    def add_token(self, token):

        if token in self._token_to_idx:
            index = self._token_to_idx[token] 
        else:
            index = len(self._token_to_idx) 
            self._token_to_idx[token] = index 
            self._idx_to_token[index] = token
            return index

    def lookup_token(self, token):
        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]
    def lookup_index(self, index):

        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index) 
        return self._idx_to_token[index]
    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)
    def __len__(self):
        return len(self._token_to_idx)





class JobVectorizer(object):
    """ The Vectorizer which coordinates the Vocabularies and puts them to use"""
    def __init__(self, text_vocab):
        """
        Args:
        text_vocab (Vocabulary): maps words to integers
        """
        self.text_vocab = text_vocab

    def vectorize(self, words):
        one_hot = np.zeros(len(self.text_vocab), dtype=np.float32) 
        for token in words:
            one_hot[self.text_vocab.lookup_token(token)] = 1
        return one_hot 

    @classmethod
    def from_dataframe(cls, df, text_column, cutoff=10):
        text_vocab = Vocabulary(add_unk=True)

        # Add top words if count > provided count
        word_counts = Counter()
        for words in df[text_column]:
            for word in words:
                word_counts[word] += 1

        for word, count in word_counts.items():
            if count > cutoff:
                text_vocab.add_token(word)

        return cls(text_vocab)


class JobDatasetOnehot(Dataset):
    def __init__(self, df, vectorizer, text_column, label_column): 
        super(JobDatasetOnehot, self).__init__()

        self.text_column = text_column
        self.label_column = label_column

        self.df = df
        self._vectorizer = vectorizer

    def __len__(self):
        return len(self.df) 

    def __getitem__(self, index):
        words = self.df.iloc[index][self.text_column]
        text_vector = self._vectorizer.vectorize(words)
        
        label = self.df[self.label_column][index]
        return {'text_vector': text_vector,
                'label': label} 

    def get_num_batches(self, batch_size):
        return len(self) // batch_size



from torch.utils.data import Dataset 

class JobDatasetMyembedding(Dataset):
    def __init__(self, df, embedding, text_column, label_column, words_len=10):
        super(JobDatasetMyembedding, self).__init__()

        self.text_column = text_column
        self.label_column = label_column
        self.df = df
        self.embedding = embedding
        self.words_len = words_len

    def __getitem__(self, index):
        
        words = self.df[self.text_column][index]
        
        if len(words) < self.words_len:
            words = words + ["<pad>"] * (self.words_len - len(words))
        words = words[:self.words_len]
        text_vector = self.embedding[words]
        label = self.df[self.label_column][index]

        return {'text_vector': text_vector,
                'label': label} 

    def __len__(self): 
        return len(self.df)


class JobDatasetPretrainembedding(Dataset):

    vocab = torchtext.vocab.pretrained_aliases["glove.6B.100d"]()
    vocab.itos.extend(['<UNK>'])
    vocab.stoi['<UNK>'] = vocab.vectors.shape[0]
    vocab.vectors = torch.cat([vocab.vectors, torch.zeros(1, vocab.dim)], dim=0)
    word_embedding = nn.Embedding.from_pretrained(vocab.vectors)


    def __init__(self, df, text_column, label_column, words_len=10):
        super(JobDatasetPretrainembedding, self).__init__()

        self.text_column = text_column
        self.label_column = label_column
        self.df = df
        self.words_len = words_len

    def __getitem__(self, index):
        
        words = self.df[self.text_column][index]
        if len(words) < self.words_len:
            words = words + ["<pad>"] * (self.words_len - len(words))
        words = words[:self.words_len]

        word_idxs = torch.tensor([self.vocab.stoi.get(w, 400000) for w in words], dtype=torch.long)
        text_vector = self.word_embedding(word_idxs)
        label = self.df[self.label_column][index]

        return {'text_vector': text_vector,
                'label': label} 

    def __len__(self): 
        return len(self.df)




def collate_fn(batch):
    batch_text_vectors = [b['text_vector'] for b in batch]
    batch_labels = [b['label'] for b in batch]


    batch_text_vectors = torch.tensor(np.stack(batch_text_vectors))
    batch_labels = torch.tensor(np.stack(batch_labels))


    batch_data = {
        'batch_text_vectors': batch_text_vectors,
        'batch_labels': batch_labels
    }

    return batch_data