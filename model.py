import numpy as np
import mlx.core as mx 
import mlx.nn as nn 


class VAE(nn.Module):
    
    def __init__(self, img_size, latent_dim, n_cond=0):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.encoder = Encoder(img_size, self.latent_dim, n_cond)
        self.decoder = Decoder(img_size, self.latent_dim, n_cond)
    
    def reparameterize(self, mean, logvar):
        if self.training:
            std = mx.exp(0.5 * logvar)
            eps = mx.random.normal(std.shape)
            return mean + std * eps
        else:
            return mean
    
    def __call__(self, x, y=None):
        latent_dist = self.encoder(x, y)
        latent_sample = self.reparameterize(*latent_dist)
        reconstruct = self.decoder(latent_sample, y)
        return reconstruct, latent_dist, latent_dist
    
    def sample_latent(self, x, y=None):
        latent_dist = self.encoder(x, y)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class Encoder(nn.Module):
    
    def __init__(self, img_size,
                 latent_dim=10,
                 n_cond=0,):

        super().__init__()
        
        # Layer parameters
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.input_dim = img_size[0]
        
        # dense layers
        self.dense1 = nn.Linear(self.input_dim, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear layers for mean and variance
        self.lin_mu = nn.Linear(hidden_dim + n_cond, self.latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim + n_cond, self.latent_dim)
        
    def __call__(self, x, y=None):
        
        # Dense layers with ReLU activations
        x = nn.relu(self.dense1(x))
        x = nn.relu(self.dense2(x))
        
        # Linear layers for mean and variance (log-var)
        if y is not None:
            mu = self.lin_mu(mx.concatenate([x, y], axis=1))
            logvar = self.lin_logvar(mx.concatenate([x, y], axis=1))
        else:
            mu = self.lin_mu(x)
            logvar = self.lin_logvar(x)
        
        return mu, logvar
    

class Decoder(nn.Module):
    
    def __init__(self, img_size,
                 latent_dim=10,
                 n_cond=0,):
        super().__init__()
        
        # Layer parameters
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.input_dim = img_size[0]
        
        # Dense layers
        self.dense1 = nn.Linear(latent_dim + n_cond, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, self.input_dim)
    
    def __call__(self, z, y=None):
        
        # Dense layers
        if y is not None:
            x = nn.relu(self.dense1(mx.concatenate([z, y], axis=1)))
        else:
            x = nn.relu(self.dense1(z))
        x = nn.relu(self.dense2(x))
        x = self.dense3(x)

        return mx.sigmoid(x)
    