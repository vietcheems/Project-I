import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch import nn
from tqdm.notebook import tqdm
from torch.autograd import Variable 
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt 

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.backends.cudnn.deterministic = True
import time
import transformers

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score


# Create a dataset class
class pmDataset(Dataset):
  def __init__(self, df, target, features, sequence_length= 24):
    self.features = features
    self.target = target
    self.sequence_length = sequence_length
    self.y = torch.tensor(df[target].values).float()
    self.X = torch.tensor(df[features].values).float()
    self.max_len = len(df[features])


  def __len__(self):
    return self.X.shape[0]

  def __getitem__(self, i): 
    if i >= self.sequence_length:
        i_start = i - self.sequence_length
        x = self.X[i_start:i, :]
    else:
        padding = self.X[0].repeat(self.sequence_length - i , 1)
        x = self.X[0:i , :]
        x = torch.cat((padding, x), 0)

    return x, self.y[i]
  
  def get_dataset(self):
    x_tensor = []
    y_tensor = []

    for i in range(self.max_len):
      x,y = self[i]
      x_tensor.append(x)
      y_tensor.append(y)
      
    x_tensor = torch.stack(x_tensor,0)
    y_tensor = torch.stack(y_tensor,0)
    
    return x_tensor, y_tensor

#Create a dataloader
def get_loader(df, batch_size = 64, shuffle = True, sequence_length = 24 ):  
  features = df.columns
  target = ['pm']
  sequence_length = sequence_length
  dataset = pmDataset(df,target = target, features = features, sequence_length = sequence_length)
  loader = DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
  )
  return loader, dataset


#Define main LSTM model
class LSTM(nn.Module):
  def __init__(self, input_length, input_size, hidden_size, num_layers, device):
    super(LSTM, self).__init__()
    self.input_length = input_length
    self.input_size = input_size #input size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.device = device
    self.lstm = nn.LSTM(
        input_size= self.input_size,
        hidden_size= self.hidden_size,
        batch_first=True,
        num_layers= self.num_layers
    )
    self.fc = nn.Linear(in_features=  self.hidden_size, out_features = 1)
    self.relu = nn.ReLU()
    torch.nn.init.xavier_uniform_(self.fc.weight)

  def forward(self, x):
    h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device) #hidden state
    c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device) #internal state
    # Propagate input through LSTM
    output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state


    hn = hn[0].view(-1, self.hidden_size) #reshaping the data for Dense layer next
    
    out = self.relu(hn)
    out = self.fc(out)   #Final Output
    return out


#Function to save and load models
def save_checkpoint(state, file_name):
  torch.save(state, file_name)

def load_checkpoint(checkpoint_path, model):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint["state_dict"], strict = False)


#Function to unscale a tensor
def unscale_tensor(sequence, scaler):
  sequence = sequence.cpu().detach().numpy()
  sequence =  np.reshape(sequence, (sequence.shape[0],1))
  padded_sequence = np.zeros(shape = (sequence.shape[0],14) )
  padded_sequence[:,0] = sequence[:,0]
  sequence = scaler.inverse_transform(padded_sequence)[:,0]
  return sequence

#Function for inference and calculate losses
def predict_model(model, df, scaler = None, print_loss = False):
  input_length, device = model.input_length, model.device
  test_loader, test_dataset = get_loader(df, batch_size = 64, shuffle = False, sequence_length = input_length)
  model.eval()
  with torch.no_grad():
    test_input, test_target = test_dataset.get_dataset()
    test_input = test_input.to(device)
    test_target = test_target.to(device)
    output = model(test_input)
    output, test_target = output.squeeze(), test_target.squeeze()
    output, test_target = unscale_tensor(output, scaler), unscale_tensor(test_target, scaler)

  if print_loss == True:
    mae = mean_absolute_error(output, test_target)
    mse = mean_squared_error(test_target, output)
    r2 = r2_score(test_target, output)
    
    print("The test losses are: ", end = " ")
    print('MSE loss : ', mse, end = " ")
    print('MAE loss : ', mae, end = " ")
    print('R2 score: ', r2)
    print()
    
  return output, test_target

#Function to plot the models' predictions
def plot_model_predictions(model, df, scaler = None):
  input_length, device = model.input_length, model.device
  test_loader, test_dataset = get_loader(df, batch_size = 64, shuffle = False, sequence_length = input_length)
  preds, targets = predict_model(model, df, scaler)

  plt.figure(figsize=(30,10))

  plt.plot(targets, label = "target")
  plt.plot(preds, label = 'prediction') 
  plt.legend(fontsize= 16)

  plt.suptitle( ("Predictions of our best LSTM model" ),fontsize = 30  ,weight = 'bold')
  plt.xlabel('Time Steps', fontsize = 15)
  plt.ylabel('PM 2.5', fontsize = 15)
