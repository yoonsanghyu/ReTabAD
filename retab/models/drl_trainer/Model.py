import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def random_orthogonal_vectors(num_vectors, vector_dim):
    # ensure linear independent
    while True:
        random_matrix = np.random.randn(num_vectors, vector_dim)
        if np.linalg.matrix_rank(random_matrix) == num_vectors:
            break
    
    # initialize
    orthogonal_vectors = np.zeros((num_vectors, vector_dim))
    
    # Gram-Schmidt process
    for i in range(num_vectors):
        v = random_matrix[i]
        
        for j in range(i):
            v -= np.dot(v, orthogonal_vectors[j]) * orthogonal_vectors[j]
        
        # normalize current vector
        orthogonal_vectors[i] = v / np.linalg.norm(v)
    
    return orthogonal_vectors


class DRL(nn.Module):
    def __init__(self, model_config):
        super(DRL, self).__init__()
        self.data_dim = model_config['data_dim']
        self.hidden_dim = model_config['hidden_dim']

        self.en_nlayers = model_config['en_nlayers']
        self.de_nlayers = model_config['de_nlayers']
        self.model_config = model_config
        
        if model_config['diversity'] == False:
            if model_config['plearn'] == False:
                self.basis_vector = nn.Parameter(torch.rand(model_config['basis_vector_num'], self.hidden_dim), requires_grad=False)
            else:
                self.basis_vector = nn.Parameter(torch.rand(model_config['basis_vector_num'], self.hidden_dim), requires_grad=True)
        else:
            if model_config['plearn'] == False:
                self.basis_vector = nn.Parameter(torch.tensor(random_orthogonal_vectors(model_config['basis_vector_num'], self.hidden_dim)).float(), requires_grad=False)
            else:
                self.basis_vector = nn.Parameter(torch.tensor(random_orthogonal_vectors(model_config['basis_vector_num'], self.hidden_dim)).float(), requires_grad=True)

        phi = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-2):
            phi.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            phi.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        phi.append(nn.Linear(encoder_dim,model_config['basis_vector_num'],bias=False))
        self.phi = nn.Sequential(*phi)

        encoder = []
        encoder_dim = self.data_dim
        for _ in range(self.en_nlayers-1):
            encoder.append(nn.Linear(encoder_dim,self.hidden_dim,bias=False))
            encoder.append(nn.LeakyReLU(0.2, inplace=True))
            encoder_dim = self.hidden_dim
        self.encoder = nn.Sequential(*encoder)

        decoder = []
        decoder.append(nn.Linear(self.hidden_dim,self.data_dim,bias=False))
        self.decoder = nn.Sequential(*decoder)

    def forward(self, x_input):
        h = self.encoder(x_input)

        weight = F.softmax(self.phi(x_input), dim=1)
        h_ = weight@self.basis_vector

        mse = F.mse_loss(h, h_, reduction='none')

        return mse.sum(dim=1,keepdim=True)
    
