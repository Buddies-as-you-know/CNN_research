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
import torchvision
import torchvision.transforms as transforms
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
        self.seq_len = 50              # 画像の Height を時系列のSequenceとしてLSTMに入力する
        self.feature_size = 432         # 画像の Width を特徴量の次元としてLSTMに入力する
        self.hidden_layer_size = 30   # 隠れ層のサイズ
        self.lstm_layers = 1           # LSTMのレイヤー数　(LSTMを何層重ねるか)
        self.conv1 = nn.Conv2d(1, 32,kernel_size=5, stride=1)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=5)
        self.lstm = nn.LSTM(self.feature_size, 
                            self.hidden_layer_size, 
                            num_layers = self.lstm_layers)
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc = nn.Linear(self.hidden_layer_size, 7)
        
    def init_hidden_cell(self, batch_size, device): # LSTMの隠れ層 hidden と記憶セル cell を初期化
        hedden = torch.zeros(self.lstm_layers, batch_size, self.hidden_layer_size).to(device)
        cell = torch.zeros(self.lstm_layers, batch_size, self.hidden_layer_size).to(device)     
        return (hedden, cell)

    def forward(self, x, device):
        """
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        """
        batch_size = x.shape[0]
        
        self.hidden_cell = self.init_hidden_cell(batch_size, device)
        
        x = x.view(batch_size, self.seq_len, self.feature_size)  # (Batch, Cannel, Height, Width) -> (Batch, Height, Width) = (Batch, Seqence, Feature)
                                                                 # 画像の Height を時系列のSequenceに、Width を特徴量の次元としてLSTMに入力する
        x = x.permute(1, 0, 2)                                   # (Batch, Seqence, Feature) -> (Seqence , Batch, Feature)
        
        lstm_out, (h_n, c_n) = self.lstm(x, self.hidden_cell)    # LSTMの入力データのShapeは(Seqence, Batch, Feature)
                                                                 # (h_n) のShapeは (num_layers, batch, hidden_size)
        x = h_n[-1,:,:]                                          # lstm_layersの最後のレイヤーを取り出す  (B, h)
        #x = self.dropout1(x)
        x = self.fc(x)
        
        return x

def train_model(net, train_data, criterion, optimizer, device, num_epochs):
    net.train()
    # epochのループ
    for epoch in range(num_epochs):
        running_loss = 0.0
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

            # 未学習時の検証性能を確かめるため、epoch=0の訓練は省略
        if (epoch == 0):
            continue

            # データローダーからミニバッチを取り出すループ
        for i , (inputs, labels) in enumerate(train_data,0):
          inputs = inputs.to(device)
          labels = labels.to(device)
          inputs = inputs.view(7,1,50,432)
          optimizer.zero_grad()
          outputs = net(inputs,device)
          loss = criterion(outputs, labels.long())
          loss.backward()
          optimizer.step()
          running_loss += loss.item()
        if i % 400 == 399:
            print('[{:5d}] loss: {:.12f}'.format(i + 1, running_loss / 400))
            running_loss = 0.0

def test_model(net,test_data,device):
    net.eval() #推論モード
    true_list = []
    pred_list = []
    test_total_acc = 0
    with torch.set_grad_enabled(False):
        for n,(data,label) in enumerate(test_data,0):
                data = data.to(device)
                label = label.to(device)
                data = data.view(7,1,50,432)
                #p1d = (1, 403,2,20)
                #data = F.pad(data, p1d, "constant", 0)
                output = net(data,device)
                test_total_acc += cal_acc(label.long(),output)
                pred = torch.argmax(output , dim =1)
                pred_list += pred.detach().cpu().numpy().tolist()
                true_list += label.detach().cpu().numpy().tolist()
                #print(pred_list)
                #print(true_list)
    print(f"test acc:{test_total_acc/len(test_data)*100}")
    cm = confusion_matrix(true_list, pred_list)
    sns.heatmap(cm, annot=True, cmap='Blues')
    plt.savefig('sklearn_confusion_matrix_annot_blues.png')

if __name__ == "__main__":
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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=0.001)
    train_model(net, train_loader, criterion, optimizer, device, num_epochs=100)
    params = net.state_dict()
    torch.save(params, "net_00.prm", pickle_protocol=4)
    test_model(net,test_loader,device)