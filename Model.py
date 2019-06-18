from __future__ import print_function
import CNN
import torch
from torch import nn, optim
from torch.autograd import Variable
import numpy as np
import tensorflow as tf

class model():
    def __init__(self, features, learning_rate=0.01, enum=1):
        # ----预定义参数----
        self.learning_rate = learning_rate
        self.enum = enum
        self.available_gpu = torch.cuda.is_available()
        #self.available_gpu = False
        # ----选择模型----
        features = torch.from_numpy(features)
        self.model = CNN.CNN(features)
        if self.available_gpu:
            self.model = self.model.cuda()
        # ----定义损失函数和优化器----
        self.criterion = nn.CrossEntropyLoss()  # 交叉熵
        # self.criterion = nn.MSELoss() # 均值误差，在此输出不适用
        # 优化器自动更新model中各层的参数，存储在parameters中；lr是学习率
        # self.optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        # self.optimizer = optim.Adadelta(self.model.parameters(), lr=self.learning_rate)
        self.optimizer = optim.Adam(self.model.parameters())
        # self.optimizer = optim.ASGD(self.model.parameters())


    def train_model(self, train_data, train_label, maxloss=1e-3):
        # 指示model进入训练模式
        self.model.train()

        running_acc = .0
        running_loss = .0
        for epoch in range(self.enum):
            running_acc = .0
            running_loss = .0

            img = train_data
            label = train_label
            # show images，显示一个batch的图像集，很有意思
            # imshow(torchvision.utils.make_grid(img))

            if self.available_gpu:
                img = img.cuda()
                label = label.cuda()
            else:
                img = Variable(img)
                label = Variable(label)

            # 需要清除已存在的梯度,否则梯度将被累加到已存在的梯度
            self.optimizer.zero_grad()
            # forward
            out = self.model(img)
            # backward
            loss = self.criterion(out, label)
            loss.backward()
            # 更新参数optimize
            self.optimizer.step()

            running_loss += float(loss.item() * label.size(0))

            # 按维度dim返回最大值及索引
            _, pred = torch.max(out, dim=1)
            current_num = (pred == label).sum()
            running_acc += float(current_num.item())


            #acc = (pred == label).float().mean()
            #print("epoch: {}/{}, loss: {:.6f}, running_acc: {:.6f}".format(epoch + 1, self.enum,
            #                                                                             loss.item(), acc.item()))

            if loss.item() < maxloss:
                #print("Warning: training break: current loss has less than maxloss!")
                break

        return running_loss / len(train_label), running_acc / len(train_label)


    def test_model(self, test_data, test_label):
        # 指示model进入测试模式
        self.model.eval()
        eval_loss = .0
        eval_acc = .0

        img = test_data
        label = test_label

        if self.available_gpu:
            img = img.cuda()
            label = label.cuda()
        else:
            img = Variable(img)
            label = Variable(label)

        out = self.model(img)
        loss = self.criterion(out, label)

        # 这里与训练不同的是，仅计算损失，不反向传播误差和计算梯度
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, dim=1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.item()

        #print('**testing loss: {:.6f}, testing accuracy: {:.6f}**'.format(
        #    eval_loss / (len(test_label)),
        #    eval_acc / (len(test_label))
        #))

        return eval_loss / (len(test_label)), eval_acc / (len(test_label))


    def predict(self, X):
        # 指示model进入测试模式
        self.model.eval()
        y = []

        img = X

        if self.available_gpu:
            img = img.cuda()
        else:
            img = Variable(img)

        out = self.model(img)
        _, pred = torch.max(out, dim=1)

        y.append(pred)
        return y


    def save_model(self, path = 'cnnmodel/cnn.pth'):
        torch.save(self.model.state_dict(), path)  # 保存模型


    def load_model(self, path = 'cnnmodel/cnn.pth'):
        self.model.load_state_dict(torch.load(path))

def train(model, features, train_feature, train_label, sentence_len, feature_len, batch_size=64, cnnmodelpath='./cnnmodel/cnn.pth'):
    k = int(len(train_feature) / batch_size)
    loss = .0
    acc = .0
    print("开始训练共%d组数据" %k)
    for i in range(k):
        curdata = []
        curlabel = []
        for j in range(batch_size * i, batch_size * (i + 1)):
            x = genxinput(features, train_feature[j], sentence_len, feature_len)
            label = train_label[j]
            # embedding之前
            #curdata.append([x])
            curdata.append(x)
            curlabel.append(label)

        data = np.array(curdata)
        # embedding之前
        #data = data.astype(np.float32)
        data = torch.from_numpy(data)
        data = data.type(torch.LongTensor)

        label = np.array(curlabel)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        loss, acc = model.train_model(data, label)

        if ((i + 1) % 100 == 0):
            print("----Batch %d/%d: loss = %.6f, acc = %.6f----" %((i+1), k, loss, acc))
            model.save_model(cnnmodelpath)
    model.save_model(cnnmodelpath)
    print("Traning complete! loss = %.6f, acc = %.6f" % (loss, acc))

    return loss, acc


def valid(model, features, valid_feature, valid_label, sentence_len, feature_len, batch_size = 640):
    k = int(len(valid_feature) / batch_size)
    avg_loss = .0
    avg_acc = .0
    print("开始验证")
    for i in range(k):
        curdata = []
        curlabel = []
        for j in range(batch_size * i, batch_size * (i + 1)):
            x = genxinput(features, valid_feature[j], sentence_len, feature_len)
            label = valid_label[j]
            # embedding之前
            # curdata.append([x])
            curdata.append(x)
            curlabel.append(label)

        data = np.array(curdata)
        # embedding之前
        # data = data.astype(np.float32)
        data = torch.from_numpy(data)
        data = data.type(torch.LongTensor)

        label = np.array(curlabel)
        label = torch.from_numpy(label)
        label = label.type(torch.LongTensor)

        loss, acc = model.test_model(data, label)
        avg_loss += loss
        avg_acc += acc

    avg_loss /= k
    avg_acc /=k
    print('****Validation loss: {:.6f}, Validation accuracy: {:.6f}****'.format(avg_loss, avg_acc))

    print("Validation complete!")

    return avg_loss, avg_acc


def predict(model, features, test_feature, sentence_len, feature_len, batch_size = 500):
    result = []
    k = int(len(test_feature) / batch_size)
    print("****开始预测****")
    for i in range(k):
        curdata = []
        for j in range(batch_size * i, batch_size * (i + 1)):
            x = genxinput(features, test_feature[j], sentence_len, feature_len)
            # embedding之前
            # curdata.append([x])
            curdata.append(x)

        data = np.array(curdata)
        # embedding之前
        # data = data.astype(np.float32)
        data = torch.from_numpy(data)
        data = data.type(torch.LongTensor)

        y = model.predict(data)
        y = y[0].cpu().numpy()
        result.extend(list(y))

    print("预测结束!")
    return result


def genxinput(features, fidx, sentence_len, feature_len):
    fealen = len(features)
    x = []
    fcount = sentence_len - len(fidx)
    for item in fidx:
        if item == -1:
            x.append(fealen-1)
        else:
            x.append(item)
    if fcount < 0:
        return x[:32]
    for i in range(fcount):
        x.append(fealen-1)
    return x
    # embedding前
    '''
    x = []
    filler = [.0] * feature_len
    count = 0
    for idx in fidx:
        if count >= sentence_len:
            break
        if idx != -1:
            x.append(features[idx])
        else:
            x.append(filler)
        count += 1
    for k in range(sentence_len - count):
        x.append(filler)
    return x
    '''