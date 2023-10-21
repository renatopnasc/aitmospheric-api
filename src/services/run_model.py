import librosa
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F

import os

model_path = 'deepLearning/model.pth'
inputs_path = 'deepLearning/latent_space/'
output_path = 'assets/audios/aitmosphere.wav'

def post_process(spectrogram):
  return librosa.griffinlim(S=np.expm1(np.maximum(spectrogram, 0)), win_length=2000, hop_length=500, random_state=0)

class ResidualLayer(nn.Module):
    """
    One residual layer inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    """

    def __init__(self, in_dim, h_dim, res_h_dim):
        super(ResidualLayer, self).__init__()
        self.res_block = nn.Sequential(
            nn.ReLU(True),
            nn.Conv2d(in_dim, res_h_dim, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.ReLU(True),
            nn.Conv2d(res_h_dim, h_dim, kernel_size=1,
                      stride=1, bias=False)
        )

    def forward(self, x):
        x = x + self.res_block(x)
        return x

class ResidualStack(nn.Module):
    """
    A stack of residual layers inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack
    """

    def __init__(self, in_dim, h_dim, res_h_dim, n_res_layers):
        super(ResidualStack, self).__init__()
        self.n_res_layers = n_res_layers
        self.stack = nn.ModuleList(
            [ResidualLayer(in_dim, h_dim, res_h_dim)]*n_res_layers)

    def forward(self, x):
        for layer in self.stack:
            x = layer(x)
        x = F.relu(x)
        return x

class Decoder(nn.Module):
    """
    This is the p_phi (x|z) network. Given a latent sample z p_phi 
    maps back to the original space z -> x.

    Inputs:
    - in_dim : the input dimension
    - h_dim : the hidden layer dimension
    - res_h_dim : the hidden dimension of the residual block
    - n_res_layers : number of layers to stack

    """

    def __init__(self, in_dim, h_dim, n_res_layers, res_h_dim):
        super(Decoder, self).__init__()
        kernel = 4
        stride = 2

        self.inverse_conv_stack = nn.Sequential(
            nn.ConvTranspose2d(
                in_dim, h_dim, kernel_size=kernel-1, stride=stride-1, padding=1),
            ResidualStack(h_dim, h_dim, res_h_dim, n_res_layers),
            nn.ConvTranspose2d(h_dim, h_dim,
                              kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim, h_dim // 2,
                              kernel_size=kernel, stride=stride, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(h_dim//2, 1, kernel_size=kernel,
                              stride=stride, padding=(1,0), output_padding=(1,0))
        )
        

    def forward(self, x):
        return self.inverse_conv_stack(x)

# Selecionar Indice
idx = np.random.randint(0, 999)

# Carregar Entrada
input = torch.load(inputs_path + str(idx) + ".pt")

# Carregar Model
model = Decoder(64, 128, 2, 32)
model.load_state_dict(torch.load(model_path))
model.eval()

# Passar o arquivo no modelo
output = model(input).squeeze(0).cpu().detach().numpy()

# Pós processamento
audio = post_process(output)

if os.path.isfile(output_path):
    os.remove(output_path)

# Cria arquivo de Áudio
sf.write(output_path, audio[0,:], 22050, 'PCM_24')
