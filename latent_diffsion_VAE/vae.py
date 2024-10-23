import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from model import Encoder, Decoder
from data_preprocess import NanoCT_Dataset
from losses import LPIPSWithDiscriminator


class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean).to(device=self.parameters.device)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape).to(device=self.parameters.device)
        return x

    def kl(self, other=None):
        if self.deterministic:
            return torch.Tensor([0.])
        else:
            if other is None:
                return 0.5 * torch.sum(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=[1, 2, 3])
            else:
                return 0.5 * torch.sum(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=[1, 2, 3])

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
    

class AutoencoderKL(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.encoder = Encoder(**vars(config))
        self.decoder = Decoder(**vars(config))
        self.loss = LPIPSWithDiscriminator(disc_start=50001, kl_weight=1e-6, disc_weight=0.5)
        self.quant_conv = torch.nn.Conv2d(2*config.z_channels, 2*config.embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(config.embed_dim, config.z_channels, 1)
        self.data_dir = config.data_dir
        self.resolution = config.resolution
        self.batch_size = config.batch_size
        self.automatic_optimization = False

        if config.ckpt_path is not None:
            self.init_from_ckpt(config.ckpt_path, ignore_keys=[])

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu")["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        self.load_state_dict(sd, strict=False)
        print(f"Restored from {path}")

    def encode(self, x):
        h = self.encoder(x)
        moments = self.quant_conv(h)
        posterior = DiagonalGaussianDistribution(moments)
        return posterior

    def decode(self, z):
        z = self.post_quant_conv(z)
        dec = self.decoder(z)
        return dec

    def forward(self, input, sample_posterior=True):
        posterior = self.encode(input)
        if sample_posterior:
            z = posterior.sample()
        else:
            z = posterior.mode()
        dec = self.decode(z)
        return dec, posterior

    def train_dataloader(self):
        trn_set = NanoCT_Dataset(self.data_dir, img_size=self.resolution)
        return DataLoader(trn_set, batch_size=self.batch_size, num_workers=4, shuffle=True, pin_memory=False, persistent_workers=True) 
    
    def get_input(self, batch):
        batch = batch.to(memory_format=torch.contiguous_format).float()
        return batch

    def training_step(self, batch, batch_idx):
        inputs = self.get_input(batch)
        reconstructions, posterior = self(inputs)
        g_opt, d_opt = self.optimizers()
        # train encoder+decoder+logvar
        aeloss, log_dict_ae = self.loss(inputs, reconstructions, posterior, 0, self.global_step,
                                        last_layer=self.get_last_layer(), split="train")
        self.log("aeloss", aeloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        g_opt.zero_grad()
        self.manual_backward(aeloss)
        g_opt.step()

        # train the discriminator
        discloss, log_dict_disc = self.loss(inputs, reconstructions, posterior, 1, self.global_step,
                                            last_layer=self.get_last_layer(), split="train")
        self.log("discloss", discloss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        d_opt.zero_grad()
        self.manual_backward(discloss)
        d_opt.step()

    def configure_optimizers(self):
        lr = self.learning_rate
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))
        return opt_ae, opt_disc

    def get_last_layer(self):
        return self.decoder.conv_out.weight
    
    @torch.no_grad()
    def log_images(self, x, only_inputs=False):
        log = dict()
        x = x.to(self.device)
        if not only_inputs:
            xrec, posterior = self(x)
            log["samples"] = self.decode(torch.randn_like(posterior.sample()))
            log["reconstructions"] = xrec
        log["inputs"] = x
        return log