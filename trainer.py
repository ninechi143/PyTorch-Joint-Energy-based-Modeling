# -*- coding: utf-8 -*-

import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import cv2      # for save images
import imageio  # for making GIF
from PIL import Image
import matplotlib.pyplot as plt

from utils.dataset import downstream_task_dataset, collate_fn
from utils.model import Downstream_Task_Model
from utils.loss import Energy_Based_LogMarginal_Loss

import os
from pathlib import Path
from time import perf_counter
from tqdm import tqdm
from datetime import datetime
import shutil


class model_trainer():

    def __init__(self,args):

        self.mode = args.mode.lower()
        self.gpu = args.gpu

        self.load_ckpt = args.load_ckpt
        self.lr = args.lr
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.optim = args.optimizer
        self.no_log = args.no_log
        self.note = args.note

        self.reinit_freq = args.reinit_freq
        self.sgld_step = args.sgld_step
        self.sgld_lr = args.sgld_lr
        self.sgld_std = args.sgld_std
        

        self.time_slot = datetime.today().strftime("%Y%m%d_%H%M")
        self.logdir = os.path.join(os.path.dirname(__file__), self.time_slot + "_logs" + self.note)
        self.ckpt_dir = os.path.join(self.logdir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok = True)
                    
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"[!] torch version: {torch.__version__}")
        print(f"[!] computation device: {self.device}, index : {self.gpu}")
        print(f"[!] execution mode: {self.mode}")
    

    def __printer(info):
        def wrap1(function):
            def wrap2(self , *args, **argv):
                print(f"[!] {info}...")
                function(self , *args, **argv)
                print(f"[!] {info} Done.")
            return wrap2
        return wrap1


    @__printer("Data Loading")
    def load_data(self):

        # mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        # std = [x / 255 for x in [63.0, 62.1, 66.7]]

        transforms_train = torchvision.transforms.Compose( [
                                    # torchvision.transforms.Lambda(lambda x: 2. * (np.array(x) / 255.) - 1.),
                                    # torchvision.transforms.Lambda(lambda x: torch.from_numpy(x).float()),
                                    # torchvision.transforms.Lambda(lambda x: torch.permute(x, (2,0,1))),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Pad(4, padding_mode="reflect"),
                                    torchvision.transforms.RandomCrop(32),
                                    torchvision.transforms.RandomHorizontalFlip(p = 0.5),
                                    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                                    # torchvision.transforms.Normalize(mean, std),
                                    torchvision.transforms.Lambda(lambda x: x + 0.03 * torch.randn_like(x))
                                ])
        
        transforms_test = torchvision.transforms.Compose( [
                                    torchvision.transforms.ToTensor(),                            
                                    torchvision.transforms.Normalize((.5, .5, .5), (.5, .5, .5)),
                                    # torchvision.transforms.Normalize(mean, std),
                                ])

        
        self.train_dataset = downstream_task_dataset(train_stage=True, transform=transforms_train)    
        self.test_dataset = downstream_task_dataset(train_stage=False, transform=transforms_test)
    

        self.train_loader = DataLoader(dataset = self.train_dataset, batch_size = self.batch_size, 
                                        shuffle = True, num_workers = 1, collate_fn = collate_fn)
        
        self.test_loader = DataLoader(dataset = self.test_dataset, batch_size = self.batch_size,
                                        shuffle = False, num_workers = 1, collate_fn = collate_fn)
        
                                                     
    @__printer("Setup")
    def setup(self):
        
        # define our model, loss function, and optimizer
        self.log_writer = None
        if self.no_log is False:
            self.log_writer = SummaryWriter(self.logdir)
            self.record_args(self.logdir)
        

        self.EnergyModel = Downstream_Task_Model(device=self.device, reinit_freq=self.reinit_freq, SGLD_step=self.sgld_step, SGLD_lr=self.sgld_lr, SGLD_std=self.sgld_std).to(self.device)
        self.show_parameter_size(self.EnergyModel, "EnergyModel")

        self.CrossEntropy = torch.nn.CrossEntropyLoss(reduction="mean").to(self.device)
        self.Energy_Based_LogMarginal_Loss = Energy_Based_LogMarginal_Loss().to(self.device)
        

        trainable_model_parameters = []
        trainable_model_parameters += self.EnergyModel.parameters()

        if self.optim.lower() == "adam":
            self.optimizer = torch.optim.Adam(trainable_model_parameters, lr=self.lr)
        elif self.optim.lower() == "rmsprop":
            self.optimizer = torch.optim.RMSprop(trainable_model_parameters, lr=self.lr)
        else:
            self.optimizer = torch.optim.SGD(trainable_model_parameters, lr=self.lr, momentum = 0.9, weight_decay = 5e-4, nesterov = True)
        
        
        def warmup_cosine_annealing(step, total_steps, lr_max, lr_min):
            warm_up_iter = 1000
            if step < warm_up_iter:
                return step / warm_up_iter
            return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
                        lr_lambda=lambda step: warmup_cosine_annealing(step, self.epochs * len(self.train_loader),
                                                                1,  1e-6 / self.lr))# since lr_lambda computes multiplicative factor

        # self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode = "min" , factor = 0.5, patience = 20,  min_lr = 1e-6)

        # load checkpoint file to resume training
        if self.load_ckpt:
            self.load()



    def execute(self):

        if self.mode == "baseline_train":
            self.train(energy_train_flag=False)
            self.save("baseline_train_end")

        elif self.mode == "energy_train":
            self.train(energy_train_flag=True)
            self.save("energy_train_end")

    

    @__printer("Model Training")
    def train(self, energy_train_flag = True):
        print("\n")

        avg_time = 0
        counter_for_logging = 0
        # evaluation metrics
        self.best_loss = float("inf")
        self.best_posterior_loss = float("inf")
        self.best_marginal_loss = float("inf")
        self.best_acc = -1e8
        self.state = {"train_loss":[], "test_loss":[], "test_acc":[], "test_posterior_loss":[], "test_marginal_loss":[]}

        for epoch in range(self.epochs):

            st = perf_counter()
            
            self.EnergyModel.train()
            train_loss = 0
            for i , batch, in tqdm(enumerate(self.train_loader), desc="Train Progress", leave=False):

                real_data, targets = batch[0].to(self.device), batch[1].to(self.device)
                
                real_logit, real_logsumexp_term = self.EnergyModel(real_data)
                
                loss = self.CrossEntropy(real_logit, targets)
                if energy_train_flag:

                    sampled_data = self.EnergyModel.sample(batch_size = real_data.shape[0])
                    sampled_logsumexp_term = self.EnergyModel.minus_energy_score(x = sampled_data)
                    loss_energy = self.Energy_Based_LogMarginal_Loss(real_logsumexp_term, sampled_logsumexp_term)
                    loss += loss_energy
                    if np.abs(loss_energy.item()) >= 1e5 or torch.isnan(loss):
                        assert False, "Training Diverge!!!"


                # updates
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()

                # evaluation
                # train_loss += (loss.item() / n_train_total_steps)
                train_loss = 0.9 * train_loss + 0.1 * loss.item()
                
                counter_for_logging += 1
                if counter_for_logging % 100 == 1:
                    self.train_logging(counter_for_logging)
                if counter_for_logging % 500 == 1:
                    self.make_long_term_SGLD_gif()

            self.state["train_loss"].append(train_loss)

            test_num = 0
            test_loss = 0
            test_posterior_loss = 0
            test_marginal_loss = 0
            test_accuracy = 0
            
            self.EnergyModel.eval()
            # with torch.no_grad(): # here we can not use it since we need to calculate grad in EBM when sampling data
            for i , batch in tqdm(enumerate(self.test_loader), desc="Test Progress", leave=False):
    
                real_data, targets = batch[0].to(self.device), batch[1].to(self.device)
                
                real_logit, real_logsumexp_term = self.EnergyModel(real_data)
                
                loss = self.CrossEntropy(real_logit, targets)
                test_posterior_loss = 0.9 * test_posterior_loss + 0.1 * loss.item()

                ## to reduce training time, we don't calculate marginal_loss in test_loder
                # if energy_train_flag:
                #     sampled_data = self.EnergyModel.sample(batch_size = real_data.shape[0])
                #     self.EnergyModel.eval()
                #     sampled_logsumexp_term = self.EnergyModel.minus_energy_score(x = sampled_data)
                #     loss_energy = self.Energy_Based_LogMarginal_Loss(real_logsumexp_term, sampled_logsumexp_term)
                #     loss += loss_energy
                #     test_marginal_loss = 0.9 * test_marginal_loss + 0.1 * loss_energy.item()


                # test_loss += (loss.item() / n_test_total_steps)
                test_loss = 0.9 * test_loss + 0.1 * loss.item()

                prediction = self.EnergyModel.posterior_predict(logit = real_logit)

                test_num += real_data.shape[0]
                test_accuracy += torch.sum(torch.argmax(prediction, dim = -1) == targets)

                
            test_accuracy = (test_accuracy / test_num).detach().cpu().numpy()
            

            self.state["test_loss"].append(test_loss)
            self.state["test_acc"].append(test_accuracy)
            self.state["test_posterior_loss"].append(test_posterior_loss)
            self.state["test_marginal_loss"].append(test_marginal_loss)

            self.log_writer.add_scalar(f"Train Loss" , train_loss , epoch)
            self.log_writer.add_scalar(f"Test Loss" , test_loss , epoch)
            self.log_writer.add_scalar(f"Test Acc." , test_accuracy , epoch)
            self.log_writer.add_scalar(f"Test Posterior Loss" , test_posterior_loss , epoch)
            self.log_writer.add_scalar(f"Test Marginal Loss" , test_marginal_loss , epoch)
        

            avg_time = avg_time + (perf_counter() - st - avg_time) / (epoch+1)
            print(f"[!] ┌── Epoch: [{epoch+1}/{self.epochs}] done, Training time per epoch: {avg_time:.3f}")
            print(f"[!] ├── Train Loss: {train_loss:.6f}")
            print(f"[!] ├── Test Loss: {test_loss:.6f}")
            print(f"[!] ├── Acc.: {test_accuracy:.4f}, Posterior Loss: {test_posterior_loss:.6f}, Marginal Loss: {test_marginal_loss:.6f}")
            print(f"[!] └──────────────────────────────────────────────────────────────\n")

            if test_loss <= self.best_loss:
                self.best_epoch = epoch
                self.best_loss = test_loss
                self.best_acc = test_accuracy
                self.best_posterior_loss = test_posterior_loss
                self.best_marginal_loss = test_marginal_loss
                # self.save(f"best_{epoch:04d}")
                self.save(f"best")

                
            if epoch % 10 == 0:
                print(f"[!] Best Epoch: {self.best_epoch}, Loss: {self.best_loss:.6f}, Acc.: {self.best_acc:.4f}, Posterior: {self.best_posterior_loss:.6f}, Marginal: {self.best_marginal_loss:.6f}\n") 

        if not self.no_log:
            self.log_writer.close()

        for k, v in self.state.items():
            plt.plot(np.arange(len(v)), np.array(v), color = "r")
            plt.title(k); plt.xlabel("epoch")
            plt.savefig(os.path.join(self.logdir, f"{k}.png")); plt.close()

        print(f"[!] Best Epoch: {self.best_epoch}, Loss: {self.best_loss:.6f}, Acc.: {self.best_acc:.4f}, Posterior: {self.best_posterior_loss:.6f}, Marginal: {self.best_marginal_loss:.6f}\n") 


    @__printer("Model Saving")    
    def save(self , name = ""):

        keys = ["EnergyModel", "replay_buffer"]
        values = [self.EnergyModel.state_dict(), self.EnergyModel.replay_buffer.detach().cpu()]

        checkpoint = {
            k:v for k, v in zip(keys, values) 
        }
        
        torch.save(checkpoint , os.path.join(self.ckpt_dir, f"ckpt_{name}.pth"))


    @__printer("Model Loading")  
    def load(self):
        checkpoint = torch.load(self.load_ckpt)
        self.EnergyModel.load_state_dict(checkpoint['EnergyModel'])
        self.EnergyModel.replay_buffer = checkpoint["replay_buffer"].to(self.device)


    def train_logging(self, step):

        if not self.no_log:

            path = os.path.join(self.logdir, "_train_logging")
            os.makedirs(path, exist_ok=True)

            self.EnergyModel.eval()            
            num = 64
            sampled_data = self.EnergyModel.sample(batch_size = num)

            torchvision.utils.save_image(torch.clamp(sampled_data, -1, 1),
                                         os.path.join(path, f"sampled_image_{step:08d}.png"),
                                         normalize=True, 
                                         nrow = int(np.sqrt(num)))

            self.log_writer.add_image("sampled data", 
                                      torchvision.utils.make_grid(torch.clamp(sampled_data, -1, 1), nrow = int(np.sqrt(num)) , normalize = True), 
                                      step)

            self.EnergyModel.train()

    @__printer("Make GIF")  
    def make_long_term_SGLD_gif(self):

        if not self.no_log:

            path = os.path.join(self.logdir, "_long_term_SGLD_demo")
            os.makedirs(path, exist_ok=True)

            self.EnergyModel.eval()            
            num = 64
            sample_process = self.EnergyModel.sample_long_term_SGLD(batch_size = num)

            for i, sample in enumerate(sample_process):
                torchvision.utils.save_image(torch.clamp(sample, -1, 1),
                                         os.path.join(path, f"sample_SGLD_step_{i:05d}.png"),
                                         normalize=True, 
                                         nrow = int(np.sqrt(num)))

            self.EnergyModel.train()
            self.make_gif(path, 15)

    def make_gif(self, path, fps = 10):

        non_sorted_list = list(Path(path).rglob("*.png"))
        if len(non_sorted_list) == 0:
            print(f"[!] No such images in folder to make GIF. Please check")
            return

        png_list = sorted(non_sorted_list)

        process = [cv2.imread(str(i))[:, :, ::-1] for i in png_list]
        imageio.mimsave(os.path.join(self.logdir , f"SGLD_process_demo.gif") , process , fps = fps)
        # [os.remove(i) for i in png_list]
        

    def record_args(self , path):

        source_code_path = os.path.dirname(__file__)
        backup_path = os.path.join(path, "src_backup")
        os.makedirs(backup_path, exist_ok=True)
        shutil.copy(os.path.join(source_code_path, "trainer.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "main.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "dataset.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "loss.py"), backup_path)
        shutil.copy(os.path.join(source_code_path, "utils", "model.py"), backup_path)

        with open(os.path.join(path , "command_args.txt") , "w") as file:
            file.write(f"mode = {self.mode}\n")
            file.write(f"gpu = {self.gpu}\n")
            file.write(f"load ckpt = {self.load_ckpt}\n")
            file.write(f"learning rate = {self.lr}\n")
            file.write(f"optimizer = {self.optim}\n")
            file.write(f"batch size = {self.batch_size}\n")
            file.write(f"epochs = {self.epochs}\n")
            file.write(f"reinit freq = {self.reinit_freq}\n")
            file.write(f"sgld step = {self.sgld_step}\n")
            file.write(f"sgld lr = {self.sgld_lr}\n")
            file.write(f"sgld std = {self.sgld_std}\n")
            
        return


    def show_parameter_size(self, model, model_name = "Model"):
        model_parameters = filter(lambda p: p.requires_grad, model.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(f"[!] {model_name} - number of parameters: {params}")
