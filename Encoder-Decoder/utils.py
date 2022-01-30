import os
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
import transformers
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
os.environ['PYTHONHASHSEED'] = str(42)

import time as time

from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import seaborn as sns
import matplotlib.pyplot as plt


#Create dataset
class pmDataset(Dataset):
    def __init__(self, df, target, features, input_length, output_length):
        self.features = features
        self.target = target
        self.input_length = input_length
        self.output_length = output_length
        self.y = torch.tensor(df[target].values).float()
        self.X = torch.tensor(df[features].values).float()
        self.max_len = len(df[features])

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        if i >= self.input_length:
            i_start = i - self.input_length
            x = self.X[i_start:i, :]
        else:
            padding = self.X[0].repeat(self.input_length - i, 1)
            x = self.X[0:i, :]
            x = torch.cat((padding, x), 0)

        targets = torch.zeros(self.output_length, 1)

        if i > (self.max_len - self.output_length):
            for j in range((self.max_len - i)):
                targets[j] = self.y[i + j]

            for j in range(self.max_len - i, self.output_length):
                targets[j] = targets[(self.max_len - i - 1)]

        else:
            targets = self.y[i: i + self.output_length]
        return x, targets

    def get_dataset(self):
        x_tensor = []
        y_tensor = []

        for i in range(self.max_len):
            x, y = self[i]
            x_tensor.append(x)
            y_tensor.append(y)

        x_tensor = torch.stack(x_tensor, 0)
        y_tensor = torch.stack(y_tensor, 0)
        return x_tensor, y_tensor


#Create dataloader
def get_loader(df, input_length, output_length,batch_size = 64,shuffle = True):
  features = df.columns[1:].tolist()
  features.append('pm')
  target = ['pm']

  dataset = pmDataset(df,target = target, features = features, input_length = input_length, output_length = output_length)
  loader = DataLoader(
      dataset=dataset,
      batch_size=batch_size,
      shuffle=shuffle,
  )
  return loader, dataset

class Attention(nn.Module):
  def __init__(self, key_size, query_size, num_hiddens, dropout, **kwargs):
    super(Attention, self).__init__(**kwargs)
    self.W_k = nn.Linear(key_size, num_hiddens, bias=False)
    self.W_q = nn.Linear(query_size, num_hiddens, bias=False)
    self.w_v = nn.Linear(num_hiddens, 1, bias=False)
    self.dropout = nn.Dropout(dropout)

  def forward(self, queries, keys, values):
    queries, keys = self.W_q(queries), self.W_k(keys)
    # After dimension expansion, shape of `queries`: (`batch_size`, no. of
    # queries, 1, `num_hiddens`) and shape of `keys`: (`batch_size`, 1,
    # no. of key-value pairs, `num_hiddens`). Sum them up with
    # broadcasting
    features = queries.unsqueeze(2) + keys.unsqueeze(1)
    features = torch.tanh(features)
    # There is only one output of `self.w_v`, so we remove the last
    # one-dimensional entry from the shape. Shape of `scores`:
    # (`batch_size`, no. of queries, no. of key-value pairs)
    scores = self.w_v(features).squeeze(-1)
    self.attention_weights = nn.functional.softmax(scores, dim=-1)
    # Shape of `values`: (`batch_size`, no. of key-value pairs, value
    # dimension)
    return torch.bmm(self.dropout(self.attention_weights), values)


class Encoder(nn.Module):
    def __init__(self, encoder_input_size, hidden_size, num_layers, device, drop_out):
        super().__init__()
        self.input_size = encoder_input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.device = device
        self.lstm = nn.LSTM(input_size=encoder_input_size, hidden_size=hidden_size, num_layers=self.num_layers,
                            batch_first=True, dropout=drop_out)
        self.init_weights()

    def forward(self, x):
        # x: (batch_size,sequence_length, input_size)
        x = x.to(self.device)
        # h_0, c_0 (num_layers, batch_size, hidden_size)
        h_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device)  # hidden state
        c_0 = torch.zeros(self.num_layers, x.shape[0], self.hidden_size).to(self.device)  # internal state

        # output: (batch_size, sequence_length, hidden_size)
        # h_n: (num_layers, batch_size, hidden_size)
        output, (hidden, cell) = self.lstm(x, (h_0, c_0))  # lstm with input, hidden, and internal state

        return output, (hidden, cell)

    def init_weights(self):
        for m in self.modules():
            if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
                for name, param in m.named_parameters():
                    if 'weight_ih' in name:
                        torch.nn.init.xavier_uniform_(param.data)
                    elif 'weight_hh' in name:
                        torch.nn.init.orthogonal_(param.data)
                    elif 'bias' in name:
                        param.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

class Decoder(nn.Module):
  def __init__(self, hidden_size, num_layers, device,
                dropout=0, output_size = 1):
    super(Decoder, self).__init__()
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.device = device
    self.dropout = dropout
    self.output_size = output_size
    self.lstm = nn.LSTM(self.hidden_size, self.hidden_size, self.num_layers,
        dropout= self.dropout, batch_first = True)
    self.fc = nn.Linear(hidden_size, output_size)
    self.relu = nn.ReLU()
    self.init_weights()

  def forward(self, x, hidden, cell):

    # Feed the decoder input, and the last hidden state and cell state through an lstm layer
    input = x.to(device)
    out, (hidden, cell) = self.lstm(input, (hidden, cell))

    # RELU and linear layer for output
    out = self.relu(out)
    out = self.fc(out)

    return out, hidden, cell


  def init_weights(self):
    for m in self.modules():
      if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in m.named_parameters():
          if 'weight_ih' in name:
            torch.nn.init.xavier_uniform_(param.data)
          elif 'weight_hh' in name:
              torch.nn.init.orthogonal_(param.data)
          elif 'bias' in name:
              param.data.fill_(0)
      elif isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


class EncoderDecoder(nn.Module):
    def __init__(self, device, encoder_input_size=33, input_length=48, output_length=12,
                 output_size=1, hidden_size=32, num_layers=1, dropout=0):
        super(EncoderDecoder, self).__init__()
        self.encoder_input_size = encoder_input_size
        self.device = device
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_length = input_length
        self.output_length = output_length
        self.encoder = Encoder(self.encoder_input_size, self.hidden_size, self.num_layers, self.device, dropout)
        self.decoder = Decoder(self.hidden_size, self.num_layers, self.device, dropout)
        self.attention = Attention(hidden_size, hidden_size, hidden_size, dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x, target=None, teacher_force_ratio=0):
        # Feed the input through the encoder, retrieve the encoded vector and hidden states, cell states
        encoded_vector, (enc_hidden, enc_cell) = self.encoder(x)
        hidden, cell = enc_hidden, enc_cell
        encoded_vector = encoded_vector.to(self.device)
        # The first input to the decoder will be the pm2.5 of time step i-1, if we were to predict from time step i
        decoder_input = x[:, :, -1:]
        decoder_input = decoder_input[:, -1:, -1:]

        # Iterate
        outputs = []
        for step in range(self.output_length):
            # decoder_input =  transform_decoder_input(decoder_input, encoded_vector.shape[1], encoded_vector.shape[2])
            decoder_input = decoder_input.to(self.device)
            # decoder_input = decoder_input + encoded_vector
            # decoder_input = torch.cat((decoder_input, encoded_vector), dim = 1)

            if hidden.shape[0] == 1:
                query = hidden.permute((1, 0, 2))
            else:
                h = torch.unsqueeze(hidden[-1], dim=0)
                query = h.permute((1, 0, 2))

            context = self.attention(query, encoded_vector, encoded_vector)

            # print(context.shape, decoder_input.shape)
            # decoder_input = context + decoder_input
            decoder_input = torch.cat((context[:, :, :-1], decoder_input), dim=2)

            decoder_output, hidden, cell = self.decoder(decoder_input, hidden, cell)

            teacher_force = random.random() < teacher_force_ratio
            if teacher_force:
                decoder_input = target[:, step + 1:step + 2, :]
            else:
                decoder_input = decoder_output

            outputs.append(decoder_output)

        outputs = torch.cat(outputs, 1)

        return outputs

#Function to save a model's weights
def save_checkpoint(state, file_name):
  torch.save(state, file_name)

#Function to load an existing model
def load_checkpoint(checkpoint_path, model):
  checkpoint = torch.load(checkpoint_path)
  model.load_state_dict(checkpoint["state_dict"], strict = False)

#Function to unscale the tensors
def unscale_tensor(tensor, scaler):
  unscaled_tensor =  np.zeros(shape = tensor.shape)
  for i in range(tensor.shape[0]):
      sequence = tensor[i]
      sequence = sequence.cpu().detach().numpy()
      sequence =  np.reshape(sequence, (sequence.shape[0],1))
      padded_sequence = np.zeros(shape = (sequence.shape[0],14) )
      padded_sequence[:,0] = sequence[:,0]
      sequence = scaler.inverse_transform(padded_sequence)[:,0]
      unscaled_tensor[i] = sequence

  return unscaled_tensor


# Function to predict and test model performance on a dataset

def predict_model(model, df, scaler = None, print_loss=False):
    inp_length, output_length, device = model.input_length, model.output_length, model.device
    test_loader, test_dataset = get_loader(df, batch_size=64, shuffle=False, input_length=inp_length,
                                           output_length=output_length)
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

        print("The test loss is: ", end=" ")
        print('MSE loss : ', mse, end=" ")
        print('MAE loss : ', mae, end=" ")
        print('R2 score: ', r2)
        print()

    return output, test_target


def time_step_performance(time_step, model, df, scaler = None):
    input_length, output_length, device = model.input_length, model.output_length, model.device
    test_loader, test_dataset = get_loader(df, batch_size=64, shuffle=False, input_length=input_length,
                                           output_length=output_length)

    preds, targets = predict_model(model, df,scaler = scaler, print_loss=False)

    preds = [preds[i][time_step - 1] for i in range(preds.shape[0])]
    targets = [targets[i][time_step - 1] for i in range(targets.shape[0])]

    mse = mean_squared_error(preds, targets)
    mae = mean_absolute_error(preds, targets)
    r2 = r2_score(preds, targets)

    return (mse, mae, r2), (preds, targets)

# Function to predict and test model performance at a particular future time step
def test_model_timesteps(time_step, model, df, scaler = None ):
    (mse, mae, r2), (preds, targets) = time_step_performance(time_step, model, df ,scaler)
    print('MSE loss : ', mse, end=" ")
    print('MAE loss : ', mae, end=" ")
    print('R2 score: ', r2)

    plt.figure(figsize=(20, 8))
    plt.plot(targets, label="target")
    plt.plot(preds, label='prediction')
    plt.legend(fontsize=16)

    plt.suptitle("Predictions at time step {0} of our best Encoder-Decoder model".format(time_step), weight='bold',
                 fontsize=30)
    plt.xlabel('Time Steps', fontsize=15)
    plt.ylabel('PM 2.5', fontsize=15)
    #plt.savefig('/content/drive/MyDrive/enc_no_ssa_step={0}'.format(time_step), bbox_inches='tight')
