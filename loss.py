import mlx.core as mx 
import mlx.nn as nn 


RECON_DIST = ["bernoulli", "laplace", "gaussian"]


def vae_loss(model, data, label=None, beta=1.0, r_dist="gaussian"):
    recon , latent_dist, _ = model(data, label)
    recon_loss = reconstruction_loss(data, recon, r_dist)
    reg_loss = kl_normal_loss(*latent_dist)
    return recon_loss + beta * reg_loss


def reconstruction_loss(data, recon, distribution="gaussian"):
    batch_size = data.shape[0]
    if distribution == "gaussian":
        loss = 0.5 * mx.sum(mx.square(data * 255 - recon * 255)) / 255
    elif distribution == "bernoulli":
        loss = mx.sum(data * mx.log(recon)) + ((1 - data) * mx.log(1 - recon))
    else:
        assert distribution not in RECON_DIST
        raise ValueError(f'Unknown distribution: {distribution}')
    return loss / batch_size


def kl_normal_loss(mean, logvar, ):
    latent_kl = 0.5 * (-1 - logvar + mean.square() + logvar.exp()).mean(axis=0)
    total_kl = latent_kl.sum()
    return total_kl
