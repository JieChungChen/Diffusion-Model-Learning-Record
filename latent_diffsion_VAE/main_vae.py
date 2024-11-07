import argparse, os, random
import torch
import torchvision
import numpy as np
from tqdm import tqdm
from PIL import Image
from torchinfo import summary
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from vae import AutoencoderKL
from data_preprocess import NanoCT_Dataset_AE


def get_args_parser():
    parser = argparse.ArgumentParser('Latent Diffusion KLVAE', add_help=False)
    parser.add_argument('--train', default=False, type=bool)
    # Path
    parser.add_argument('--data_dir', default='./training_data_n', type=str)
    # parser.add_argument('--ckpt_path', default=None, type=str) 
    parser.add_argument('--ckpt_path', default='./checkpoints/klvae_190.ckpt', type=str) 
    # Training 
    parser.add_argument('--batch_size', default=1, type=int)
    parser.add_argument('--accumulate_grad_steps', default=8, type=int)
    parser.add_argument('--epoch', default=200, type=int)
    parser.add_argument('--lr', default=4.5e-6, type=float)
    parser.add_argument('--grad_clip', default=1., type=float)
    parser.add_argument('--disc_start', default=20001, type=int)
    # Model
    parser.add_argument('--vae_inference', default=False, type=bool)
    parser.add_argument('--resolution', default=512, type=int)
    parser.add_argument('--z_channels', default=3, type=int)
    parser.add_argument('--embed_dim', default=3, type=int)
    parser.add_argument('--ch', default=128, type=int)
    parser.add_argument('--ch_mult', default=[1, 2, 2, 4], type=list)
    parser.add_argument('--num_res_blocks', default=2, type=int)
    return parser


def main(args):
    model = AutoencoderKL(args).cuda()
    model.train()
    # summary(model, input_size=(args.batch_size, 1, 512, 512))

    if args.train:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        os.makedirs('checkpoints', exist_ok=True)
        torch.save(model.state_dict(), 'checkpoints/ckpt_0.ckpt')

        # Dataset
        trn_set = NanoCT_Dataset_AE(args.data_dir)
        trn_loader = DataLoader(trn_set, batch_size=args.batch_size, num_workers=0, shuffle=True, pin_memory=True) 

        # Optimizer
        lr = args.accumulate_grad_steps * torch.cuda.device_count() * args.batch_size * args.lr
        opt_ae = torch.optim.Adam(list(model.encoder.parameters())+
                                  list(model.decoder.parameters())+
                                  list(model.quant_conv.parameters())+
                                  list(model.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(model.loss.discriminator.parameters(),
                                    lr=lr, betas=(0.5, 0.9))

        # Training
        step = 0
        for e in range(args.epoch):
            with tqdm(trn_loader, dynamic_ncols=True) as tqdmDataLoader:
                for img in tqdmDataLoader:
                    step+=1
                    img = img.float().cuda()
                    reconstructions, posterior = model(img)

                    # update autoencoders
                    aeloss = model.loss(img, reconstructions, posterior, 0, step, last_layer=model.get_last_layer())
                    aeloss.backward() 
                    if step % args.accumulate_grad_steps == 0:
                        opt_ae.step()
                        opt_ae.zero_grad(set_to_none=True)
                    
                    # update discriminator
                    discloss = model.loss(img, reconstructions, posterior, 1, step, last_layer=model.get_last_layer())
                    discloss.backward()
                    if step % args.accumulate_grad_steps == 0:
                        opt_disc.step()
                        opt_disc.zero_grad(set_to_none=True)

                    tqdmDataLoader.set_postfix(ordered_dict={
                        "epoch": e,
                        "aeloss: ": aeloss.item(),
                        "discloss: ": discloss.item(),
                        "img shape: ": img.shape,
                        "LR": opt_ae.state_dict()['param_groups'][0]["lr"]
                    })

                if (e+1)%10==0:
                    torch.save(model.state_dict(), 'checkpoints/klvae_%d.ckpt'%(e+1))

    else:
        model.eval()
        size = args.resolution
        # dataset = NanoCT_Dataset_AE(args.data_dir)
        test_path = './valid_data_n/data2/gt_dref/'
        test_folders = sorted(os.listdir(test_path))
        for i, folder in enumerate(test_folders):
            val_files = [_ for _ in os.listdir(test_path+folder+'/') if _.endswith('tif')]
            random.seed(2024)
            rnd_id = random.randint(0, len(val_files)-1)
            raw_img = Image.open('./valid_data_n/data2/original/%s/%s'%(folder, val_files[rnd_id]))
            raw_img = np.array(raw_img)
            dref_truth = Image.open('./valid_data_n/data2/gt_dref/%s/%s'%(folder, val_files[rnd_id]))
            dref_truth = np.array(dref_truth)
            ref_truth = (raw_img/dref_truth)/raw_img.max()
            input_img = torch.Tensor(raw_img/raw_img.max())
            outputs = model.encode(input_img.cuda().float().view(1, 1, size, size)).mode()
            outputs = model.decode(outputs).detach().cpu()
            psnr = -10*torch.log10(((input_img-outputs)**2).mean())

            plt.subplot(121)
            plt.title('input img')
            plt.axis('off')
            plt.imshow(input_img, cmap='gray')
            plt.subplot(122)
            plt.title('recon. PSNR=%.2f'%psnr)
            plt.axis('off')
            plt.tight_layout()
            plt.imshow(input_img, cmap='gray')
            plt.savefig('fig/result_%d'%i)
            plt.close()


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)