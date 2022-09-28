# from msilib import sequence
import os
import gensim
from collections import Counter
import json
import torch

from torch.utils.data import Dataset, DataLoader
import numpy as np

sequence_length = 20

#setup Glove word embeddings
with open('glove.6B/glove.6B.50d.txt', 'r') as f:
    lines = f.readlines()

glove = dict()

for line in lines:
    items = line.split()
    word = items[0]
    vector = np.array(list(map(float, items[1:])))
    glove[word] = vector

def build_vocab(path, min_word_count = 20):
    counter = Counter()

    with open(path, 'r') as f:
        lines = f.readlines()
    processedLines = [gensim.utils.simple_preprocess(sentence,min_len=1) for sentence in lines]
    for line in processedLines:
        counter.update(line)
    #initialise a dictionary or look up table
    word2id = {}
    id2word = {}
    word2id['<pad>'] = 0
    word2id['<unk>'] = 1
    id2word[0] = '<pad>'
    id2word[1] = '<unk>'

    # include only those in dictionary which have occered more than min word count in the entire data.
    words = [word for word, count in counter.items() if count>min_word_count]

    for i, word in enumerate(words):
        word2id[word] = i+2
        id2word[i+2] = word

    print("Dictionary Formed and saved. The length of dictionary is-: ", len(word2id))
    return word2id, id2word





class Dataset_seq(Dataset):
	def __init__(self, word2id,id2word, train_path):
		self.word2id = word2id
		self.id2word = id2word
		self.train_path = train_path
		# read the data and label 
		self.word2representation = {}
		for word in self.word2id:
			if word in glove:
				self.word2representation[word] = glove[word]
			else:
				self.word2representation[word] = np.random.normal(scale=0.6, size=(50, ))
			# self.word2representation[word] = glove[word] if word in glove else self.word2id['<unk>']
		

		with open(train_path, 'r') as f:
			lines = f.readlines()
		processedLines = [gensim.utils.simple_preprocess(sentence,min_len=1) for sentence in lines]
		self.data = processedLines
		self.X = []
		self.Y = []
		for sent in self.data:

			if len(sent) > sequence_length:
				for j in range(len(sent)//sequence_length):
					self.X.append(sent[j*sequence_length:(j+1)*sequence_length])

			elif len(sent) < sequence_length:
				self.X.append(sent + (sequence_length - len(sent)) * ['<pad>'])
		for i in range(len(self.X)):
			self.X[i] = [word if word in self.word2id else '<unk>' for word in self.X[i]]
		self.X1,self.Y1 = self.data_l()
	def data_l(self):
		# convert the words to representations
		# convert the labels to ids
		# return the data and labels
		ans = []
		id = []
		for sent in self.X:
			ans.append([self.word2representation[word] for word in sent])
			id.append([self.word2id[word] for word in sent])
		return ans,id

	def __getitem__(self, index):
		# return the seq and label 
		return (np.array(self.X1[index], dtype=float), np.array(self.Y1[index]))

	def __len__(self):
		return(len(self.data))



