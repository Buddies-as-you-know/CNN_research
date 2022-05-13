import numpy as np
import math
from sklearn.manifold import TSNE
from matplotlib import pyplot
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import torch
def preprocessing(floatArray):
    window = np.hamming(125)
    w = np.empty(125)
    d2 = np.empty((floatArray.shape[0],27*16))
    for i in range(floatArray.shape[0]-125):
        for j in range(15):
            w = np.abs(np.fft.fftn(floatArray[i:i+125,j]*window))
            d2[i,27*j:27*j+26] = np.log10(1 + w[4:30])
    return d2
def labeler():
  label = []
  #print(array.shape[0])
  #data = np.loadtxt("/content/comand.txt",dtype='str')
  with open("C:\\Users\\owner\\Desktop\\EEGdata\\commandtrain.txt", "r") as tf:
    lines = tf.read().split(' ')
  #print(len(lines))
  for command in lines:
    for i in range(100):
      #print(command,label)
      if command == 'normal':
        label += [0]
      elif command == 'forward':
        label += [1]
      elif command == 'rightturn':
        label += [2]
      elif command == 'leftturn':
        label += [3]
      elif command == 'back':
        label += [4]
      elif command == 'up':
        label += [5]
      else:
        label += [6]
  return label

def dataset(url):
  data = np.loadtxt(url,dtype='str',delimiter=",",skiprows=5)
  b = data[:,1:17]
  floatArray = b.astype(float)
  d2 = np.empty((floatArray.shape[0],27*16))
  d2 = preprocessing(floatArray)
  ts = d2[125*10-1:125*40*7+125*10-1,:]
  return ts
def connection(data1,data2,data3,data4):
  p = np.vstack((data1,data2))
  p = np.vstack((p,data3))
  ts = np.vstack((p,data4))
  return ts
  
if __name__ == "__main__":
    data1 = dataset("C:\\Users\\owner\\Desktop\\EEGdata\\datasample\\7command.txt")
    data2 = dataset("C:\\Users\\owner\\Desktop\\EEGdata\\datasample\\7commandchoudo.txt")
    data3 = dataset("C:\\Users\\owner\\Desktop\\EEGdata\\datasample\\10byouokure.txt")
    data4 = dataset("C:\\Users\\owner\\Desktop\\EEGdata\\datasample\\12byouokure1.txt")
    ts = connection(data1,data2,data3,data4)
    label = labeler()
    train_label =np.array(label).reshape((-1, 1))
    """"
    train_data= torch.Tensor(ts)
    #test_x = torch.Tensor(test_x)
    train_data = train_data.view(2800,50,432)
    train_label = np.array(label)
    train_data = train_data.to('cpu').detach().numpy().copy()
    """
    np.savetxt('train_data.txt',ts)
    np.savetxt('train_label.txt',train_label)
    