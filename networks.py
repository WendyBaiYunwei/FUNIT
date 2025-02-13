"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import numpy as np

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.distributions as dist

from blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock

import math
from utils import sim

def assign_adain_params(adain_params, model):
    # assign the adain_params to the AdaIN layers in model
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            mean = adain_params[:, :m.num_features]
            std = adain_params[:, m.num_features:2*m.num_features]
            m.bias = mean.contiguous().view(-1)
            m.weight = std.contiguous().view(-1)
            if adain_params.size(1) > 2*m.num_features:
                adain_params = adain_params[:, 2*m.num_features:]


def get_num_adain_params(model):
    # return the number of AdaIN parameters needed by the model
    num_adain_params = 0
    for m in model.modules():
        if m.__class__.__name__ == "AdaptiveInstanceNorm2d":
            num_adain_params += 2*m.num_features
    return num_adain_params


class GPPatchMcResDis(nn.Module):
    def __init__(self, hp):
        super(GPPatchMcResDis, self).__init__()
        assert hp['n_res_blks'] % 2 == 0, 'n_res_blk must be multiples of 2'
        self.n_layers = hp['n_res_blks'] // 2
        nf = hp['nf']
        cnn_f = [Conv2dBlock(3, nf, 7, 1, 3,
                             pad_type='reflect',
                             norm='none',
                             activation='none')]
        for i in range(self.n_layers - 1):
            nf_out = np.min([nf * 2, 1024])
            cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
            cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
            cnn_f += [nn.ReflectionPad2d(1)]
            cnn_f += [nn.AvgPool2d(kernel_size=3, stride=2)]
            nf = np.min([nf * 2, 1024])
        nf_out = np.min([nf * 2, 1024])
        cnn_f += [ActFirstResBlock(nf, nf, None, 'lrelu', 'none')]
        cnn_f += [ActFirstResBlock(nf, nf_out, None, 'lrelu', 'none')]
        cnn_c = [Conv2dBlock(nf_out, 1, 1, 1, #hp['num_classes']
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)



    def forward(self, x, counterpart=None, original=None, challenge=None):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        index = torch.LongTensor(range(out.size(0))).cuda()
        resp = out[index, 0, :, :]
        # print(out.shape)
        feat = feat.mean((2,3))
        if counterpart == None:
            return resp, None, feat
        else:
            counterpart = self.cnn_f(counterpart).mean((2,3))
            original = self.cnn_f(original).mean((2,3))
            challenge = self.cnn_f(challenge).mean((2,3))
            pos_sim = sim(feat, counterpart)
            neg_sim1 = sim(feat, challenge)
            neg_sim2 = sim(counterpart, challenge)
            neg_sim3 = sim(original, challenge)
            new_out = torch.log(pos_sim / (pos_sim + neg_sim1 + neg_sim2 + neg_sim3))
        return resp, new_out, feat # new_out: batchsize, 1

    def calc_dis_fake_loss(self, input_fake):
        resp_fake, sim_score, gan_feat = self.forward(input_fake)
        # print(resp_fake)
        # print(resp_fake.shape)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        # print(total_count)
        # exit()
        fake_loss = torch.nn.ReLU()(1.0 + resp_fake).mean()
        correct_count = (resp_fake < 0).sum()
        fake_accuracy = correct_count.type_as(fake_loss) / total_count
        return fake_loss, fake_accuracy, resp_fake

    def calc_dis_real_loss(self, input_real):
        resp_real, sim_score, gan_feat = self.forward(input_real)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label, counterpart, original, challenge):
        resp_fake, sim_score, gan_feat = self.forward(input_fake, counterpart, original, challenge)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        loss = -sim_score.mean() + (-resp_fake.mean())
        correct_count = (resp_fake >= 0).sum()
        accuracy = correct_count.type_as(loss) / total_count
        # accuracy = loss
        return loss, accuracy, gan_feat

    def calc_grad2(self, d_out, x_in):
        batch_size = x_in.size(0)
        grad_dout = autograd.grad(outputs=d_out.mean(),
                                  inputs=x_in,
                                  create_graph=True,
                                  retain_graph=True,
                                  only_inputs=True)[0]
        grad_dout2 = grad_dout.pow(2)
        assert (grad_dout2.size() == x_in.size())
        reg = grad_dout2.sum()/batch_size
        return reg


class FewShotGen(nn.Module):
    def __init__(self, hp):
        super(FewShotGen, self).__init__()
        nf = hp['nf']
        nf_mlp = hp['nf_mlp']
        down_class = hp['n_downs_class']
        down_content = hp['n_downs_content']
        n_mlp_blks = hp['n_mlp_blks']
        n_res_blks = hp['n_res_blks']
        latent_dim = hp['latent_dim']
        self.enc_class_model = ClassModelEncoder(down_class,
                                                 3,
                                                 nf,
                                                 latent_dim,
                                                 norm='none',
                                                 activ='relu',
                                                 pad_type='reflect')

        self.enc_content = ContentEncoder(down_content,
                                          n_res_blks,
                                          3,
                                          nf,
                                          'in',
                                          activ='relu',
                                          pad_type='reflect')

        self.dec = Decoder(down_content,
                           n_res_blks,
                           self.enc_content.output_dim,
                           3,
                           res_norm='adain',
                           activ='relu',
                           pad_type='reflect')

        self.mlp = MLP(latent_dim,
                       get_num_adain_params(self.dec),
                       nf_mlp,
                       n_mlp_blks,
                       norm='none',
                       activ='relu')

    def forward(self, one_image, model_set):
        # reconstruct an image
        content, model_codes = self.encode(one_image, model_set)
        model_code = torch.mean(model_codes, dim=0).unsqueeze(0)
        images_trans = self.decode(content, model_code)
        return images_trans

    def encode(self, one_image, model_set):
        # extract content code from the input image
        content = self.enc_content(one_image)
        # extract model code from the images in the model set
        class_codes = self.enc_class_model(model_set)
        class_code = torch.mean(class_codes, dim=0).unsqueeze(0)
        return content, class_code

    def decode(self, content, model_code):
        # decode content and style codes to an image
        adain_params = self.mlp(model_code)
        assign_adain_params(adain_params, self.dec)
        images = self.dec(content)
        return images


class ClassModelEncoder(nn.Module):
    def __init__(self, downs, ind_im, dim, latent_dim, norm, activ, pad_type):
        super(ClassModelEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(ind_im, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(2):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        for i in range(downs - 2):
            self.model += [Conv2dBlock(dim, dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
        self.model += [nn.AdaptiveAvgPool2d(1)]
        self.model += [nn.Conv2d(dim, latent_dim, 1, 1, 0)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class ContentEncoder(nn.Module):
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type):
        super(ContentEncoder, self).__init__()
        self.model = []
        self.model += [Conv2dBlock(input_dim, dim, 7, 1, 3,
                                   norm=norm,
                                   activation=activ,
                                   pad_type=pad_type)]
        for i in range(downs):
            self.model += [Conv2dBlock(dim, 2 * dim, 4, 2, 1,
                                       norm=norm,
                                       activation=activ,
                                       pad_type=pad_type)]
            dim *= 2
        self.model += [ResBlocks(n_res, dim,
                                 norm=norm,
                                 activation=activ,
                                 pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)
        self.output_dim = dim

    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    def __init__(self, ups, n_res, dim, out_dim, res_norm, activ, pad_type):
        super(Decoder, self).__init__()

        self.model = []
        self.model += [ResBlocks(n_res, dim, res_norm,
                                 activ, pad_type=pad_type)]
        for i in range(ups):
            self.model += [nn.Upsample(scale_factor=2),
                           Conv2dBlock(dim, dim // 2, 5, 1, 2,
                                       norm='in',
                                       activation=activ,
                                       pad_type=pad_type)]
            dim //= 2
        self.model += [Conv2dBlock(dim, out_dim, 7, 1, 3,
                                   norm='none',
                                   activation='tanh',
                                   pad_type=pad_type)]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, dim, n_blk, norm, activ):

        super(MLP, self).__init__()
        self.model = []
        self.model += [LinearBlock(in_dim, dim, norm=norm, activation=activ)]
        for i in range(n_blk - 2):
            self.model += [LinearBlock(dim, dim, norm=norm, activation=activ)]
        self.model += [LinearBlock(dim, out_dim,
                                   norm='none', activation='none')]
        self.model = nn.Sequential(*self.model)

    def forward(self, x):
        return self.model(x.view(x.size(0), -1))

# class GaussianBlurLayer(nn.Module):
#     def __init__(self, kernel_size=3, channels=1):
#         super(GaussianBlurLayer, self).__init__()
#         self.kernel_size = kernel_size
#         self.channels = channels
        
#         # Learnable parameters for mean and log variance
#         self.mean = nn.Parameter(torch.randn(1), requires_grad=True)
#         self.log_var = nn.Parameter(torch.full((1,), 4.0), requires_grad=True)
#         self.to_grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1)])

#     def get_gaussian_kernel(self, kernel_size, mean, log_var, sigma=1.0):
#         x = torch.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1, dtype=torch.float32).cuda()
#         std = torch.exp(0.5 * log_var)
#         kernel = torch.exp(-0.5 * ((x - mean) / (sigma * std))**2)
#         kernel = kernel / kernel.sum()
#         return kernel

#     def forward(self, x):
#         x = self.to_grayscale(x)
#         print(x.shape)
        
#         # Get the Gaussian kernel based on the mean and log variance
#         gaussian_kernel = self.get_gaussian_kernel(self.kernel_size, self.mean, self.log_var)
        
#         # Apply Gaussian blur using the learned kernel
#         blurred = F.conv2d(x, gaussian_kernel.view(1, 1, -1, 1), padding=(self.kernel_size // 2, self.kernel_size // 2))
#         print(blurred.shape)
#         exit()

#         blurred = blurred.repeat(1, 3, 1, 1)
#         return blurred

class GaussianBlurLayer(nn.Module):
    def __init__(self, kernel_size=3, channels=1):
        super(GaussianBlurLayer, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        
        # Learnable parameters for mean and log variance for each channel
        # self.mean = nn.Parameter(torch.randn(channels), requires_grad=True)
        self.variance = nn.Parameter(torch.full((1,), 0.1), requires_grad=True)
        self.to_grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # gaussian_dist = dist.Normal(0, 1)
        # self.weights = gaussian_dist.sample((kernel_size,)).cuda()
        # self.weights = torch.cat(self.weights.sort())
        # self.weights += torch.abs(self.weights[0]) + 1.0

    def get_gaussian_kernel(self, kernel_size):
        # self.weights *= self.variance[0].item()
        # weights = self.weights

        # gaussian_dist = dist.Normal(100, self.variance)
        # weights = gaussian_dist.sample((kernel_size,)).cuda()
        # weights = torch.cat(weights.sort())
        # weights[weights > 200] = 200
        # weights[weights < 0] = 0
        
        # kernel_weights = torch.empty(kernel_size, kernel_size).cuda()
        # for i in range(0, math.ceil(kernel_size / 2)):
        #     kernel_weights[i, i:kernel_size-i] = weights[i]
        #     kernel_weights[kernel_size - i - 1, i:kernel_size-i] = weights[i]
        #     for j in range(i + 1, kernel_size - i):
        #         kernel_weights[j, i] = weights[i]
        #         kernel_weights[j, kernel_size-i-1] = weights[i]
        # kernel_weights = kernel_weights.repeat
        kernel_weights = torch.full((kernel_size, kernel_size), 1.0).cuda()
        return kernel_weights

    def forward(self, x):
        x = self.to_grayscale(x)
        # Get the Gaussian kernel based on the mean and log variance
        kernel_weights = self.get_gaussian_kernel(self.kernel_size)
        
        # # Calculate padding to keep the input and output sizes the same
        # padding = self.kernel_size // 2

        # Apply Gaussian blur using the learned kernel
        blurred = F.conv2d(x, kernel_weights.unsqueeze(0).unsqueeze(0), padding=(self.kernel_size // 2, self.kernel_size // 2))
        blurred = blurred.repeat(1, 3, 1, 1)

        return blurred