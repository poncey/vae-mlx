import numpy as np
import mlx.core as mx 
import mlx.nn as nn 


class VAE(nn.Module):
    
    def __init__(self, img_size, latent_dim, n_cond=0):
        super(VAE, self).__init__()
        
        self.latent_dim = latent_dim
        self.img_size = img_size
        self.num_pixels = self.img_size[1] * self.img_size[2]
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
        return reconstruct, latent_dist, latent_sample
    
    def sample_latent(self, x, y=None):
        latent_dist = self.encoder(x, y)
        latent_sample = self.reparameterize(*latent_dist)
        return latent_sample


class Encoder(nn.Module):
    
    def __init__(self, img_size,
                 latent_dim=10,
                 n_cond=0):

        super(Encoder, self).__init__()
        
        # Layer parameters
        hid_channels = 32
        kernel_size = 4
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start transpose convs
        self.dense_dim = (kernel_size, kernel_size, hid_channels)
        n_chan = self.img_size[-1]
        
        # Convolutional Layers
        cnn_kwargs = dict(stride=2, padding=1)
        self.conv1 = nn.Conv2d(n_chan, hid_channels, kernel_size, **cnn_kwargs)
        self.conv2 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        self.conv3 = nn.Conv2d(hid_channels, hid_channels, kernel_size, stride=2, padding=2)
        # If input image is 64x64 do fourth convolution
        if self.img_size[0] == self.img_size[1] == 64:
            self.conv_64 = nn.Conv2d(hid_channels, hid_channels, kernel_size, **cnn_kwargs)
        
        # dense layers
        self.dense1 = nn.Linear(np.product(self.dense_dim), hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        
        # Linear layers for mean and variance
        self.lin_mu = nn.Linear(hidden_dim + n_cond, self.latent_dim)
        self.lin_logvar = nn.Linear(hidden_dim + n_cond, self.latent_dim)
        
    def __call__(self, x, y=None):
        batch_size = x.shape[0]
        
        # Convolutional Layers with ReLU activations
        x = nn.relu(self.conv1(x))
        x = nn.relu(self.conv2(x))
        x = nn.relu(self.conv3(x))
        if self.img_size[0] == self.img_size[1] == 64:
            x = nn.relu(self.conv_64(x))
        
        # Dense layers with ReLU activations
        x = x.reshape(batch_size, -1)
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
                 n_cond=0):
        super(Decoder, self).__init__()
        
        # Layer parameters
        hid_channels = 32
        kernel_size = 3
        hidden_dim = 256
        self.latent_dim = latent_dim
        self.img_size = img_size
        # Shape required to start upsampling convs
        self.dense_dim = ((kernel_size + 1), (kernel_size + 1), hid_channels)
        n_chan = self.img_size[-1]
        self.img_size = img_size
        
        # Dense layers
        self.dense1 = nn.Linear(latent_dim + n_cond, hidden_dim)
        self.dense2 = nn.Linear(hidden_dim, hidden_dim)
        self.dense3 = nn.Linear(hidden_dim, np.product(self.img_size))
    
    def __call__(self, z, y=None):
        batch_size = z.shape[0]
        
        # Dense layers
        if y is not None:
            x = nn.relu(self.dense1(mx.concatenate([z, y], axis=1)))
        else:
            x = nn.relu(self.dense1(z))
        x = nn.relu(self.dense2(x))
        x = self.dense3(x)

        return mx.sigmoid(x).reshape(batch_size, *self.img_size)
    