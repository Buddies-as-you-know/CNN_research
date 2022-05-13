import numpy as np
import math
from sklearn.manifold import TSNE
from matplotlib import pyplot
import pickle
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader, Dataset
import seaborn as sns
import matplotlib.pyplot as plt
#import tqdm
from torch.autograd import Variable
from sklearn.metrics import confusion_matrix
def cal_acc(t,p):
    p_arg = torch.argmax(p,dim=1)
    return torch.sum(t == p_arg)
class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 60,kernel_size=(1, 15),stride=(1,3))
        self.conv2 = nn.Conv2d(60, 60, kernel_size=(1, 4),stride=(1,2))
        self.conv3 = nn.Conv2d(60, 60, kernel_size=(30,1),stride=(1,3))
        self.conv4 = nn.Conv2d(60, 90, kernel_size=(1, 3),stride=(1,1))
        self.conv5 = nn.Conv2d(90, 120, kernel_size=(1, 1),stride=(1,1))
        self.pool = nn.MaxPool2d(kernel_size=(1, 2),stride=(1,2))
        #lf.soft = nn.Softmax(dim = 1)
        self.fc=nn.Linear(2520,7)

    def forward(self, x):
      x = self.conv1(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = self.conv2(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = self.conv3(x)
      #print(x.shape)
      x = self.conv4(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = self.conv5(x)
      #print(x.shape)
      x = self.pool(x)
      #print(x.shape)
      x = x.view(7,2520)
      x = self.fc(x)
      #print(x.shape)
      #x = self.soft(x)
      return x

if __name__ == "__main__":
    #train_data make
    train_data = np.loadtxt("C:\\Users\\owner\\Desktop\\EEGdata\\train_data.txt",dtype='float')
    train_label = np.loadtxt("C:\\Users\\owner\\Desktop\\EEGdata\\train_label.txt",dtype='int')
    #test data make
    
    test_data = np.loadtxt("C:\\Users\\owner\\Desktop\\EEGdata\\test_data.txt",dtype='float')
    test_label = np.loadtxt("C:\\Users\\owner\\Desktop\\EEGdata\\test_label.txt",dtype='int')
    #Tensor change
    train_data=torch.Tensor(train_data)
    train_label=torch.Tensor(train_label)
    test_data=torch.Tensor(test_data)
    test_label=torch.Tensor(test_label)
    # form change
    train_data = train_data.view(100*7*4,50,432)
    test_data = test_data.view(100*7,50,432)
    # data to dat5aset
    train_dataset = TensorDataset(train_data, train_label)
    test_dataset = TensorDataset(test_data, test_label)
    train_loader = DataLoader(train_dataset, batch_size=7, shuffle=True)
    test_loader = DataLoader(test_dataset,batch_size=7,shuffle=False)
    net = Net()

# loss関数の定義
    criterion = nn.CrossEntropyLoss()
    using_cuda = torch.cuda.is_available()
    accuracies = []
# 最適化関数の定義
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(100):
      running_loss = 0.0
      for i, (inputs, labels) in enumerate(train_loader, 0):
        # zero the parameter gradients
        inputs = inputs.to(device)
        labels = labels.to(device)
        inputs = inputs.view(7,1,50,432)
        #print(inputs.shape)
        optimizer.zero_grad()
        # forward + backward + optimiz
        outputs = net(inputs)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 400 == 399:
            print('[{:d}, {:5d}] loss: {:.12f}'
                    .format(epoch + 1, i + 1, running_loss / 400))
            running_loss = 0.0

    print('Finished Training')

    # 次回から学習しなくていいように、学習済みモデルのパラメータを"net_00.prmとして保存"
    params = net.state_dict()
    torch.save(params, "net_00.prm", pickle_protocol=4)

    print('strat test ')

    ### 学習済みモデルのテスト ###
    test_total_acc = 0
    net.eval()
    #net_path = 'model.pth'
    #net.load_state_dict(torch.load(net_path))
    pred_list = []
    true_list = []
    
with torch.no_grad():
    for n,(data,label) in enumerate(test_loader):
        data = data.to(device)
        label = label.to(device)
        data = data.view(7,1,50,432)
        output = net(data)
        test_total_acc += cal_acc(label.long(),output)
        pred = torch.argmax(output , dim =1)
        pred_list += pred.detach().cpu().numpy().tolist()
        true_list += label.detach().cpu().numpy().tolist()
        #print(pred_list)
        #print(true_list)
print(f"test acc:{test_total_acc/len(test_dataset)*100}")
cm = confusion_matrix(true_list, pred_list)
sns.heatmap(cm, annot=True, cmap='Blues')
plt.savefig('sklearn_confusion_matrix_annot_blues.png')