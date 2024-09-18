# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.nn.init as init
from tqdm import tqdm


class Downstream_Task_Model(nn.Module):

    def __init__(self, device = None, reinit_freq = 0.05, SGLD_lr = 1., SGLD_std = 0.01, SGLD_step = 20):

        super().__init__()

        # self.main = nn.Sequential(
        #         nn.Conv2d(3, 16, kernel_size=kernel_size, stride=1, padding=padding), nn.LeakyReLU(0.1),
        #         nn.Conv2d(16, 32, kernel_size=kernel_size, stride=1, padding=padding), nn.LeakyReLU(0.1), nn.MaxPool2d(2, 2),
        #         nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=padding), nn.LeakyReLU(0.1),
        #         nn.AdaptiveAvgPool2d((1,1)),
        #     )        
        # self.linear = nn.Linear(64 , 10)
        
        self.input_channels = 3
        self.num_classes = 10

        self.resnet = Wide_ResNet(input_channels=self.input_channels, num_classes=self.num_classes, dropout_rate = 0.25)
        self.linear = nn.Linear(self.resnet.last_dim , self.num_classes)

        self.main = nn.Sequential(
            self.resnet, 
            self.linear,
        )
        self.softmax = nn.Softmax(dim = -1)

        # self.initialize()

        self.device = device
        self.SGLD_lr = SGLD_lr
        self.SGLD_std = SGLD_std
        self.SGLD_step = SGLD_step
        self.reinit_freq = reinit_freq
        self.buffer_size = 10000

        self.replay_buffer = torch.FloatTensor(self.buffer_size, self.input_channels, 32, 32).uniform_(-1, 1).to(self.device)

    def initialize(self):
        c = 0
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                init.normal_(m.weight, mean = 0.0 , std=0.01)
                init.zeros_(m.bias)
                c += 1
        print(f"model initialization, #modules: {c}")


    def energy_gradient(self, x):
        
        self.main.eval()

        xi = torch.FloatTensor(x.detach().cpu().data).to(self.device)
        xi.requires_grad = True

        # calculate the gradient
        logsumexp_term = self.minus_energy_score(xi)
        xi_grad = torch.autograd.grad(logsumexp_term.sum(), [xi], retain_graph=True)[0]

        self.main.train()

        return xi_grad

    def Langevin_Dynamics_step(self, x_old, SGLD_lr = None, SGLD_std = None):
        # Calculate gradient wrt x_old
        if SGLD_lr is None: SGLD_lr = self.SGLD_lr
        if SGLD_std is None: SGLD_std = self.SGLD_std

        energy_grad = self.energy_gradient(x_old)
        noise = SGLD_std * torch.randn_like(energy_grad).to(self.device)
        x_new = x_old + SGLD_lr * energy_grad + noise
        return x_new

    def sample(self, batch_size=64, x=None):

        buffer_size = len(self.replay_buffer)
        indexes = torch.randint(0, buffer_size, (batch_size,))
        buffer_sample = self.replay_buffer[indexes]
        random_sample = torch.FloatTensor(batch_size, self.input_channels, 32, 32).uniform_(-1, 1).to(self.device)

        choose_which = (torch.rand(batch_size) < self.reinit_freq).float()[:, None, None, None].to(self.device)
        x_sample = choose_which * random_sample + (1 - choose_which) * buffer_sample

        for _ in range(self.SGLD_step):
            x_sample = self.Langevin_Dynamics_step(x_sample)

        x_sample = x_sample.detach()
        
        # update replay buffer
        if len(self.replay_buffer) > 0:
            self.replay_buffer[indexes] = x_sample

        return x_sample

    def sample_long_term_SGLD(self, batch_size = 64, step_in_each_sigma = 30, step_lr_base = 1e-2,
                              sigma_init = 1, sigma_end = 0.1, sigma_seq_len = 10):

        """
        the default training setting is, SGLD_lr = 1 and SGLD_std = 0.01 
        in the inference stage, 
        even though we need to gradually decrease the "updating step" of SGLD for a better convergence,
        the ratio of lr/std should keep the same as training setting for the sake of sampling diversity,
        to this end, we'd better use the ratio of 1:0.01
        instead of roughly using the stretegy from NCSN (Y. Song, 2019).
        """
        ratio = self.SGLD_std / self.SGLD_lr

        log_seq = np.linspace(np.log(sigma_init), np.log(sigma_end), sigma_seq_len)
        sigmas = np.exp(log_seq)

        sample = torch.FloatTensor(batch_size, self.input_channels, 32, 32).uniform_(-1, 1).to(self.device)
        sample_process = [sample.detach().cpu()]
        for sigma in tqdm(sigmas, desc = f"sampling by using sigma seq.", leave=False):
            
            step_size = step_lr_base * (sigma / sigmas[-1])**2

            for _ in range(step_in_each_sigma):
                # sample = self.Langevin_Dynamics_step(sample, step_size, np.sqrt(2 * step_size)) # noise scale is too large to converge
                sample = self.Langevin_Dynamics_step(sample, step_size, ratio * step_size)
                sample = torch.clamp(sample, -1, 1)
                sample_process.append(sample.detach().cpu())

        return sample_process


    def __forward(self, x):
        x = self.main(x)
        return x

    def posterior_predict(self, x = None, logit = None):
        if logit is None:
            assert x is not None
            logit = self.__forward(x)
        prediction = self.softmax(logit)
        return prediction
    
    def minus_energy_score(self, x = None, logit = None):
        if logit is None:
            assert x is not None
            logit = self.__forward(x)
        minus_energy = torch.logsumexp(logit, dim = -1) # noting that LogSumExpY is also the term, "-E", in EBM
        return minus_energy


    def forward(self, x):    
        logit = self.__forward(x)

        # prediction = self.posterior_predict(logit = logit)
        prediction = logit  # By default, we will use built-in torch.CrossEntropy Class,
                            # which has already integrate the logSoftmax and NLL-Loss
                            # hence, here we don't need to manually apply the softmax function
                            # when only in the inference stage and you want to get a probabilistic prediction
                            # you can call self.posterior_predicit(*) to get a softmax output
        LogSumExpY_logit = self.minus_energy_score(logit = logit)
        return prediction, LogSumExpY_logit





def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True)

def conv_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.xavier_uniform(m.weight, gain=np.sqrt(2))
        init.constant(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        init.constant(m.weight, 1)
        init.constant(m.bias, 0)


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
    def forward(self, x):
        return x


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, dropout_rate, stride=1, norm=None, leak=.2):
        super(wide_basic, self).__init__()
        self.lrelu = nn.LeakyReLU(leak)
        self.bn1 = get_norm(in_planes, norm)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=True)
        self.dropout = Identity() if dropout_rate == 0.0 else nn.Dropout(p=dropout_rate)
        self.bn2 = get_norm(planes, norm)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=True),
            )

    def forward(self, x):
        out = self.dropout(self.conv1(self.lrelu(self.bn1(x))))
        out = self.conv2(self.lrelu(self.bn2(out)))
        out += self.shortcut(x)

        return out


def get_norm(n_filters, norm):
    if norm is None:
        return Identity()
    elif norm == "batch":
        return nn.BatchNorm2d(n_filters, momentum=0.9)
    elif norm == "instance":
        return nn.InstanceNorm2d(n_filters, affine=True)
    elif norm == "layer":
        return nn.GroupNorm(1, n_filters)


class Wide_ResNet(nn.Module):
    def __init__(self, depth = 28, widen_factor = 10, num_classes=10, input_channels=3,
                 sum_pool=False, norm=None, leak=.2, dropout_rate=0.0):
        super(Wide_ResNet, self).__init__()
        self.leak = leak
        self.in_planes = 16
        self.sum_pool = sum_pool
        self.norm = norm
        self.lrelu = nn.LeakyReLU(leak)

        assert ((depth-4)%6 ==0), 'Wide-resnet depth should be 6n+4'
        n = (depth-4)//6
        k = widen_factor

        print('[!] Wide-Resnet %dx%d' %(depth, k))
        nStages = [16, 16*k, 32*k, 64*k]

        self.conv1 = conv3x3(input_channels, nStages[0])
        self.layer1 = self._wide_layer(wide_basic, nStages[1], n, dropout_rate, stride=1)
        self.layer2 = self._wide_layer(wide_basic, nStages[2], n, dropout_rate, stride=2)
        self.layer3 = self._wide_layer(wide_basic, nStages[3], n, dropout_rate, stride=2)
        self.bn1 = get_norm(nStages[3], self.norm)
        self.last_dim = nStages[3]
        self.linear = nn.Linear(nStages[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, dropout_rate, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, dropout_rate, stride, norm=self.norm))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x, vx=None):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.lrelu(self.bn1(out))
        if self.sum_pool:
            out = out.view(out.size(0), out.size(1), -1).sum(2)
        else:
            out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        # out = self.linear(out)
        return out



if __name__ == "__main__":

    # encoder = _Encoder()
    # decoder = _Decoder()

    # # unit-test
    # a_batch_of_images = torch.randn(7 , 1 , 175 , 215)
    # code = encoder(a_batch_of_images)
    # decoded_image = decoder(code)
    # print(code.shape, decoded_image.shape)

    # # Gradient unit-test
    # loss = torch.mean(
    #             torch.sum(
    #                 torch.square(decoded_image - a_batch_of_images) , dim = (1,2,3)
    #             )
    #         )
    # loss.backward()
    # print(loss.item())


    # downstream_task_model unit-test
    task_model = Downstream_Task_Model()
    a_batch_of_images = torch.randn(17, 3, 32, 32)
    
    prediction = task_model.posterior_predict(a_batch_of_images)
    print(prediction.shape)

    energy_score = task_model.energy_score(a_batch_of_images)
    print(energy_score.shape)