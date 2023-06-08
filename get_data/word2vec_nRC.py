import numpy as np
from gensim.models import KeyedVectors
import pandas as pd
from itertools import product
import gensim
from gensim.models import word2vec as wv
import itertools
from gensim.models import Word2Vec
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv

train_path = "F:/桌面/凯n/ncDLRES Program/Ten_Fold_Data/nRC_training_set.xlsx"
test_path = "F:/桌面/凯n/ncDLRES Program/Ten_Fold_Data/nRC_test_set.xlsx"
all_path = "F:/桌面/凯n/ncDLRES Program/Ten_Fold_Data/ALL_nRC.xlsx"
Word_model_path = "F:/桌面/凯n/ncDLRES Program/Ten_Fold_Data/word2vec_nRC.txt"


# def getRNASequence_3kmer():
#     RNA_Sequence = []
#     data = pd.read_excel(all_path)
#     data = data.values
#     for i in range(len(data)):
#         RNA_Sequence.append(data[i][2])
#     RNA_Sequence = [x.strip() for x in RNA_Sequence]
#     k = 3  # k-mer的长度
#     alphabet = ['A', 'C', 'G', 'T']
#     kmers = [''.join(p) for p in itertools.product(alphabet, repeat=k)]
#     sentences = []
#     for sequence in RNA_Sequence:
#         sentence = []
#         for i in range(len(sequence) - k + 1):
#             kmer = sequence[i:i + k]
#             if kmer in kmers:
#                 sentence.append(kmer)
#         sentences.append(sentence)
#     return sentences
#
#
# model = Word2Vec(getRNASequence_3kmer(), sg=0, vector_size=100, window=5, min_count=1, workers=4, epochs=10)
# model.wv.save_word2vec_format(Word_model_path, binary=False)  # 将GloVe模型保存到文件中
model = KeyedVectors.load_word2vec_format(Word_model_path, binary=False)

def train_data():
    data = pd.read_excel(train_path)
    data1 = data.values
    Train_Matrix_word2vec = []

    Train_label = []
    for i in range(len(data1)):
        if data1[i][3] == "5S_rRNA":
            Train_label.append(0)
        if data1[i][3] == "5_8S_rRNA":
            Train_label.append(1)
        if data1[i][3] == "tRNA":
            Train_label.append(2)
        if data1[i][3] == "ribozyme":
            Train_label.append(3)
        if data1[i][3] == "CD-box":
            Train_label.append(4)
        if data1[i][3] == "miRNA":
            Train_label.append(5)
        if data1[i][3] == "Intron_gpI":
            Train_label.append(6)
        if data1[i][3] == "Intron_gpII":
            Train_label.append(7)
        if data1[i][3] == "HACA-box":
            Train_label.append(8)
        if data1[i][3] == "riboswitch":
            Train_label.append(9)
        if data1[i][3] == "IRES":
            Train_label.append(10)
        if data1[i][3] == "leader":
            Train_label.append(11)
        if data1[i][3] == "scaRNA":
            Train_label.append(12)
        line = data1[i][2].strip()
        allowed_chars = set("AGCT")
        line = "".join(c for c in line if c in allowed_chars)
        kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
        # 训练Word2vec模型
        embedded_sequence = np.array([model[kmer] for kmer in kmers])
        Train_Matrix_word2vec.append(embedded_sequence)

    Train_Matrix_word2vec = np.array(Train_Matrix_word2vec)
    Train_label = np.array(Train_label)
    return Train_Matrix_word2vec, Train_label

def test_data():
    data = pd.read_excel(test_path)
    data1 = data.values
    Test_Matrix_word2vec = []
    Test_label = []
    for i in range(len(data)):
        if data1[i][3] == "5S_rRNA":
            Test_label.append(0)
        if data1[i][3] == "5_8S_rRNA":
            Test_label.append(1)
        if data1[i][3] == "tRNA":
            Test_label.append(2)
        if data1[i][3] == "ribozyme":
            Test_label.append(3)
        if data1[i][3] == "CD-box":
            Test_label.append(4)
        if data1[i][3] == "miRNA":
            Test_label.append(5)
        if data1[i][3] == "Intron_gpI":
            Test_label.append(6)
        if data1[i][3] == "Intron_gpII":
            Test_label.append(7)
        if data1[i][3] == "HACA-box":
            Test_label.append(8)
        if data1[i][3] == "riboswitch":
            Test_label.append(9)
        if data1[i][3] == "IRES":
            Test_label.append(10)
        if data1[i][3] == "leader":
            Test_label.append(11)
        if data1[i][3] == "scaRNA":
            Test_label.append(12)
        line = data1[i][2].strip()
        allowed_chars = set("AGCT")
        line = "".join(c for c in line if c in allowed_chars)
        kmers = [line[i:i + 3] for i in range(0, len(line) - 2)]
        # 训练Word2vec模型
        embedded_sequence = np.array([model[kmer] for kmer in kmers])
        Test_Matrix_word2vec.append(embedded_sequence)


    Test_Matrix_word2vec = np.array(Test_Matrix_word2vec)
    Test_label = np.array(Test_label)
    return Test_Matrix_word2vec, Test_label
