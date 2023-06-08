import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import nn, optim
import numpy as np
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn.utils.rnn import pad_sequence
import ConvNeXt
from get_data import word2vec_nRC

from torch.nn import utils as nn_utils

PATH_Model = 'duibi/best_model_onehot'

# best_model_2_GRU只用了两层GRU无LSTM
class MinimalDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        data = self.data[index]
        label = self.label[index]
        return data, label

    def __len__(self):
        return len(self.data)


def collate_fn(batch_data):
    batch_data.sort(key=lambda xi: len(xi[0]), reverse=True)  # 将传进来的32个rna序列按照序列长度从大到小排序
    data_length = [len(xi[0]) for xi in batch_data]  # 存放32个RNA的序列长度
    sent_seq = [xi[0] for xi in batch_data]  # 存放32个RNA序列的每一个序列
    label = [xi[1] for xi in batch_data]  # 存放32个RNA序列的类别
    padden_sent_seq = pad_sequence([torch.from_numpy(x) for x in sent_seq], batch_first=True,
                                   padding_value=0)  # 将所有RNA序列填充到32个RNA序列里面最长的长度
    return padden_sent_seq, data_length, torch.tensor(label, dtype=torch.float32)


Train_Data, Train_Label = word2vec_nRC.train_data()
Test_Data, Test_Label = word2vec_nRC.test_data()
# print(Train_Data)
# print(len(Train_Data))
# model = Model.RNAProfileModel(Model.Residual_Block, [2, 2, 2, 2])
# model = GRU_fenlei.Bi_GRU()
model = ConvNeXt.convnext_large(13)
# model = no_attention.convnext_large(13)
if torch.cuda.is_available():
    model = model.cuda()
train_data = MinimalDataset(Train_Data, Train_Label)
test_data = MinimalDataset(Test_Data, Test_Label)

criterion = nn.CrossEntropyLoss()
optimer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.00005)
data_loader = DataLoader(train_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
# print(data_loader)
# 用于将train_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数
data_loader_test = DataLoader(test_data, batch_size=32, shuffle=True, collate_fn=collate_fn)
# 用于将test_data数据划分批次，每个批次包含32个RNA序列，划分依据collate_fn函数
model.eval()  # 首先进行预测没有进行训练模型
max_acc = 0
with torch.no_grad():
    correct = 0
    total = 0
    loss_totall = 0
    iii = 0
    for item_train in data_loader:
        train_data, train_length, train_label = item_train
        num = train_label.shape[0]
        train_data = train_data.float()
        train_label = train_label.long()
        train_data = Variable(train_data)  # 将train_data里的数据变成Variable形式，用于反向传播
        train_label = Variable(train_label)
        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
        # 将该批次所有RNA序列里面的碱基提取出来，就是一个解填充过程
        outputs = model(pack)
        loss = criterion(outputs, train_label)
        loss_totall += loss.data.item()
        iii += 1
        _, pred_acc = torch.max(outputs.data, 1)
        correct += (pred_acc == train_label).sum()
        total += train_label.size(0)
    print('Accuracy of the train Data:{}%'.format(100 * correct / total))
    print('Loss of the train Data:{}%'.format(loss_totall / iii))
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    loss_totall = 0
    iii = 0
    for item_train in data_loader_test:
        train_data, train_length, train_label = item_train
        num = train_label.shape[0]
        train_data = train_data.float()
        train_label = train_label.long()
        train_data = Variable(train_data)
        train_label = Variable(train_label)
        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
        outputs = model(pack)
        loss = criterion(outputs, train_label)
        loss_totall += loss.data.item()
        iii += 1
        _, pred_acc = torch.max(outputs.data, 1)
        correct += (pred_acc == train_label).sum()
        total += train_label.size(0)
    print('Accuracy of the test Data:{}%'.format(100 * correct / total))
    print('Loss of the test Data:{}%'.format(loss_totall / iii))

for j in range(200):
    i = 0
    model.train()
    for item_train in data_loader:
        i += 1
        train_data, train_length, train_label = item_train
        num = train_label.shape[0]
        train_data = train_data.float()
        train_label = train_label.long()
        train_data = Variable(train_data)
        train_label = Variable(train_label)
        if torch.cuda.is_available():
            train_data = train_data.cuda()
            train_label = train_label.cuda()
        pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
        outputs = model(pack)
        _, pred_acc = torch.max(outputs.data, 1)
        correct = (pred_acc == train_label).sum()
        loss = criterion(outputs, train_label)
        optimer.zero_grad()
        loss.backward()
        optimer.step()  # 模型更新
        if (i % 10 == 0 or i == 178):
            print(('Epoch:[{}/{}], Step[{}/{}], loss:{:.4f}, Accuracy:{:.4f}'.format(j + 1, 100, i, 100,
                                                                                     loss.data.item(),
                                                                                     100 * correct / num)))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_totall = 0
        iii = 0
        for item_train in data_loader:
            train_data, train_length, train_label = item_train
            num = train_label.shape[0]
            train_data = train_data.float()
            train_label = train_label.long()
            train_data = Variable(train_data)
            train_label = Variable(train_label)
            if torch.cuda.is_available():
                train_data = train_data.cuda()
                train_label = train_label.cuda()
            pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
            outputs = model(pack)
            loss = criterion(outputs, train_label)
            loss_totall += loss.data.sum()
            iii += 1
            _, pred_acc = torch.max(outputs.data, 1)
            correct += (pred_acc == train_label).sum()
            total += train_label.size(0)
        print('Accuracy of the train Data:{}%'.format(100 * correct / total))
        print('Loss of the train Data:{}%'.format(loss_totall / iii))
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_totall = 0
        iii = 0
        for item_train in data_loader_test:
            train_data, train_length, train_label = item_train
            num = train_label.shape[0]
            train_data = train_data.float()
            train_label = train_label.long()
            train_data = Variable(train_data)
            train_label = Variable(train_label)
            if torch.cuda.is_available():
                train_data = train_data.cuda()
                train_label = train_label.cuda()
            pack = nn_utils.rnn.pack_padded_sequence(train_data, train_length, batch_first=True)
            outputs = model(pack)
            loss = criterion(outputs, train_label)
            loss_totall += loss.data.item()
            iii += 1
            _, pred_acc = torch.max(outputs.data, 1)
            correct += (pred_acc == train_label).sum()
            total += train_label.size(0)
        print('Accuracy of the test Data:{}%'.format(100 * correct / total))
        print('Loss of the test Data:{}%'.format(loss_totall / iii))
        if (100 * correct / total > max_acc):
            max_acc = 100 * correct / total
            # torch.save(model, PATH_Model)
    print('maxacc:{}%'.format(max_acc))
