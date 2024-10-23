import argparse, os, random
import torch
import torchvision
import numpy as np
from PIL import Image
from torchinfo import summary
from pytorch_lightning.trainer import Trainer

from vae import AutoencoderKL


def get_args_parser():
    parser = argparse.ArgumentParser('Latent Diffusion VAE', add_help=False)
    parser.add_argument('--train', default=True, type=bool)
    # Path
    parser.add_argument('--data_dir', default='./training_data_n', type=str)
    parser.add_argument('--model_save_dir', default='./checkpoints', type=str)
    parser.add_argument('--ckpt_path', default=None, type=str) 
    # parser.add_argument('--ckpt_path', default='./lightning_logs/checkpoints/epoch=159-step=25280.ckpt', type=str) 
    # Training 
    parser.add_argument('--ngpu', default=1, type=int)
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--accumulate_grad_steps', default=1, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('--lr', default=4.5e-6, type=float)
    parser.add_argument('--grad_clip', default=1., type=float)
    # Model
    parser.add_argument('--resolution', default=512, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--z_channels', default=4, type=int)
    parser.add_argument('--embed_dim', default=4, type=int)
    parser.add_argument('--ch', default=128, type=int)
    parser.add_argument('--ch_mult', default=[1, 2, 4, 4], type=list)
    parser.add_argument('--num_res_blocks', default=2, type=int)
    return parser


def main(args):
    model = AutoencoderKL(args)
    summary(model, input_size=(4, 1, 512, 512))

    if args.train:
        model.learning_rate = args.accumulate_grad_steps * args.ngpu * args.batch_size * args.lr
        strategy = 'ddp' if args.ngpu>1 else 'auto'
        trainer = Trainer(accelerator='gpu', strategy=strategy, devices=args.ngpu, reload_dataloaders_every_n_epochs=5, 
                          num_sanity_val_steps=0, limit_val_batches=0, log_every_n_steps=100, benchmark=True)
        trainer.fit(model)
    else:
        size = args.resolution
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
            outputs = model.log_images(input_img.view(1, 1, size, size))
            torchvision.utils.save_image(outputs["inputs"], 'input.png', normalize=True)
            torchvision.utils.save_image(outputs["reconstructions"], 'reconstructions.png', normalize=True)


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)