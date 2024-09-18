# -*- coding: utf-8 -*-

import torch
import os
import argparse

from trainer import model_trainer

"""
Energy-based Out-of-Distribution Detection

paper ref.
https://proceedings.neurips.cc/paper/2020/file/f5496252609c43eb8a3d147ab9b9c006-Paper.pdf

source code ref.
https://github.com/wetliu/energy_ood
"""

def parse_args():

    parser = argparse.ArgumentParser(description='the trainer')

    # necessary argument
    parser.add_argument('--mode', default = "energy_train", choices=('baseline_train', 'energy_train', 'test'), help = "specify the execution mode")
    parser.add_argument('--gpu' , type=str , default="4", help = "choose the used gpu index")

    parser.add_argument("--load_ckpt" , type=str , default = None , help = "if you want to load existing pretrained model, setup the .pth file name.")   
    
    # for --mode train
    parser.add_argument("-l" , "--lr" , type = float , default=1e-4)
    parser.add_argument('--batch_size' , type=int , default=128)
    parser.add_argument('--epochs' , type=int , default=200)
    parser.add_argument('--optimizer' , type=str , default='adam')
    parser.add_argument("--reinit_freq" , type = float , default=0.05,  help = "reinitialization frequency when running Langevin Dynamics")
    parser.add_argument("--sgld_step" , type = int , default=20,  help = "sgld step")
    parser.add_argument("--sgld_lr" , type = float , default=1.,  help = "sgld lr")
    parser.add_argument("--sgld_std" , type = float , default=0.01,  help = "sgld std")
    parser.add_argument("--no_log" , action="store_true" , default=False , help = "True if you don't want to use Tensorboard for the logging.")
    parser.add_argument('--note' , type=str , default="")
 
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    # print(args.epochs)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    trainer = model_trainer(args)

    trainer.load_data()  # prepare dataset and dataLoader
    trainer.setup()      # define our model, loss function, and optimizer
    trainer.execute()      # execute the pipeline according to self.mode

    print("\nDone.\n")