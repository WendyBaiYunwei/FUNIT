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

from model.FUNIT.blocks import LinearBlock, Conv2dBlock, ResBlocks, ActFirstResBlock

import math
from model.FUNIT.utils import sim

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
        cnn_c = [Conv2dBlock(nf_out, 10, 1, 1, #hp['num_classes']
                             norm='none',
                             activation='lrelu',
                             activation_first=True)]
        self.cnn_f = nn.Sequential(*cnn_f)
        self.cnn_c = nn.Sequential(*cnn_c)
        self.classifier = nn.Sequential(ContentEncoder(downs=3,
                                    n_res=2,
                                    input_dim=3,
                                    dim=64,
                                    norm='in',
                                    activ='relu',
                                    pad_type='reflect',
                                    get_mean=True),
                                    nn.Linear(512, 61))
        state_dict = torch.load('/home/nus/Documents/research/augment/code/FEAT/Traffic-Translator-Pre/0.05_0.1_[75, 150, 300]/checkpoint.pth')['state_dict']
        new_dict = {}
        for k in self.classifier.state_dict().keys():
            new_k = '.'.join(k.split('.')[1:])
            if new_k != 'weight' and new_k != 'bias':
                new_k = 'encoder.' + new_k
            else:
                new_k = 'fc.' + new_k
            new_dict[k] = state_dict[new_k]
        self.classifier.load_state_dict(new_dict)
        for params in self.classifier:
            params.requires_grad = False
        
        self.pretrained_encoder = nn.Sequential(ContentEncoder(downs=3,
            n_res=2,
            input_dim=3,
            dim=64,
            norm='in',
            activ='relu',
            pad_type='reflect',
            get_mean=True),
            nn.Linear(512, 61))
        new_dict = {}
        for k in self.pretrained_encoder.state_dict().keys():
            new_k = '.'.join(k.split('.')[1:])
            if new_k != 'weight' and new_k != 'bias':
                new_k = 'encoder.' + new_k
            else:
                new_k = 'fc.' + new_k
            new_dict[k] = state_dict[new_k]
        self.pretrained_encoder.load_state_dict(new_dict)
        for params in self.pretrained_encoder:
            params.requires_grad = False


    def forward(self, x, counterpart=None, original=None, challenge=None, selector=False):
        feat = self.cnn_f(x)
        out = self.cnn_c(feat)
        index = torch.LongTensor(range(out.size(0))).cuda()
        resp = torch.tanh(out[index, :, :, :]).mean((1,2,3))
        if counterpart == None: # return feature and disc output only
            return resp
        else:
            # input_feat = self.pretrained_encoder(x)
            # counterpart = self.pretrained_encoder(counterpart)
            # idxs = torch.randperm(len(counterpart)).cuda()
            # counterpart = torch.index_select(input_feat, 0, idxs)
            # # original = self.pretrained_encoder(original)
            # challenge = self.pretrained_encoder(challenge)
            # # print(input_feat.shape, counterpart.shape)
            # # exit()
            # pos_sim1 = sim(input_feat, counterpart)
            # class_sim = pos_sim1
            # neg_sim1 = sim(input_feat, challenge)
            # neg_sim2 = sim(counterpart2, challenge)
            # neg_sim2 = sim(original, challenge)
            # print(pos_sim1, pos_sim2, neg_sim1, neg_sim2)
            # class_sim = torch.log((pos_sim1 + 1e-5) / (pos_sim1 + neg_sim1 + 1e-5))
            # primary_sim = torch.log(pos_sim1 + 1e-5) * 10
        
            if selector == True: # selector score
                return class_sim.mean() + torch.nn.ReLU()(1.0 + resp).mean() ## slack
            else:
                return resp, None, feat

    def calc_dis_fake_loss(self, input_fake):
        resp_fake = self.forward(input_fake)
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
        resp_real = self.forward(input_real)
        total_count = torch.tensor(np.prod(resp_real.size()),
                                   dtype=torch.float).cuda()
        real_loss = torch.nn.ReLU()(1.0 - resp_real).mean()
        correct_count = (resp_real >= 0).sum()
        real_accuracy = correct_count.type_as(real_loss) / total_count
        return real_loss, real_accuracy, resp_real

    def calc_gen_loss(self, input_fake, input_fake_label, counterpart, original, challenge):
        resp_fake, class_sim, gan_feat = self.forward(input_fake, counterpart, original, challenge)
        total_count = torch.tensor(np.prod(resp_fake.size()),
                                   dtype=torch.float).cuda()
        # class_loss = -class_sim.mean()
        correct_count = (resp_fake >= 0).sum()
        resp_fake = -resp_fake.mean()
        loss = resp_fake # right class loss + adversarial loss
        accuracy = correct_count.type_as(loss) / total_count
        # accuracy = loss
        # print(-class_loss.mean(), (-resp_fake.mean()))
        # exit()
        
        return resp_fake, None, accuracy, gan_feat

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

    def calc_contrast_loss(self, input_fake, content_label):
        y_ = self.classifier(input_fake)[:, content_label[0].item()]
        y_ = torch.tanh(y_) + 1.0
        return y_.mean()


class LearnableAffineTransformation(nn.Module):
    def __init__(self, input_channels, output_channels):
        super(LearnableAffineTransformation, self).__init__()
        
        # Define learnable parameters for the affine transformation
        self.rotation = nn.Parameter(torch.full((1,), 0.), requires_grad=True)
        self.translation = nn.Parameter(torch.full((2,), 0.), requires_grad=True)
        self.scaling = nn.Parameter(torch.full((2,), 0.), requires_grad=True)
        self.shearing = nn.Parameter(torch.full((2,), 0.), requires_grad=True)
        
    def forward(self, x):
        batch_size, num_channels, height, width = x.size()
        zero = torch.zeros(1).cuda()
        one = torch.ones(1).cuda()
        # Compute the batch of affine transformation matrices
        last = [zero, zero, one]
        rotation_matrix = torch.stack([torch.cos(self.rotation).reshape(1), -torch.sin(self.rotation).reshape(1),zero,
                                     torch.sin(self.rotation).reshape(1), torch.cos(self.rotation).reshape(1), zero]+last).view(3, 3).cuda()
        scaling_matrix = torch.stack([self.scaling[0].reshape(1), zero, zero, zero, self.scaling[1].reshape(1), zero]+last).view(3, 3).cuda()
        shearing_matrix = torch.stack([one, self.shearing[0].reshape(1), zero, self.shearing[0].reshape(1), one, zero]+last).view(3, 3).cuda()
        translation_matrix = torch.stack([torch.ones(1).cuda(), zero, self.translation[0].reshape(1), zero, one, self.translation[1].reshape(1)]+last).view(3, 3).cuda()
        
        affine_matrix = scaling_matrix @ shearing_matrix @ rotation_matrix @ translation_matrix
        affine_matrix = affine_matrix[:2, :]
        # Expand the batch of affine matrices
        affine_matrix = affine_matrix.unsqueeze(0).expand(batch_size, -1, -1).cuda()
        
        # Apply the affine transformation to the input
        grid = torch.nn.functional.affine_grid(affine_matrix, x.size())
        x_transformed = torch.nn.functional.grid_sample(x, grid, align_corners = True)
        
        return x_transformed


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
        # state_dict = torch.load('/home/nus/Documents/research/augment/code/FEAT/Traffic-Translator-Pre/0.05_0.1_[75, 150, 300]/checkpoint.pth')
        # new_dict = {}
        # for k in self.enc_content.state_dict().keys(): # repeated
        #     new_k = k
        #     if 'encoder.' in k:
        #         new_k = 'encoder.' + k
        #     new_dict[k] = state_dict[new_k]
        # self.enc_content.load_state_dict(new_dict)
        # self.attention = nn.Transformer(
        #     d_model=256,
        #     nhead=4,
        #     num_encoder_layers=2,
        #     dim_feedforward=256
        # ).encoder
        self.affine = LearnableAffineTransformation(3, 3).cuda()
        

        state_dict = torch.load('/home/nus/Documents/research/augment/code/FEAT/Traffic-Translator-Pre/0.05_0.1_[75, 150, 300]/checkpoint.pth')['state_dict']
        new_dict = {}
        for k in self.enc_content.state_dict().keys():
            new_dict[k] = state_dict['encoder.' + k]
        self.enc_content.load_state_dict(new_dict)
        for params in self.enc_content.parameters():
            params.requires_grad = False

        self.erase = GaussianBlurLayer()

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
    def __init__(self, downs, n_res, input_dim, dim, norm, activ, pad_type, get_mean=False):
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
        self.get_mean = get_mean

    def forward(self, x):
        if self.get_mean == True:
            x = self.model(x)
            x = torch.mean(x, (2,3))
            return x
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
    def __init__(self, kernel_size=19, channels=1):
        super(GaussianBlurLayer, self).__init__()
        self.kernel_size = kernel_size
        self.channels = channels
        
        # Learnable parameters for mean and log variance for each channel
        # self.mean = nn.Parameter(torch.randn(channels), requires_grad=True)
        # self.variance = nn.Parameter(torch.full((1,), 0.1), requires_grad=True)
        self.to_grayscale = transforms.Compose([transforms.Grayscale(num_output_channels=1)])
        # gaussian_dist = dist.Normal(0, 1)
        # self.weights = gaussian_dist.sample((kernel_size,)).cuda()
        # self.weights = torch.cat(self.weights.sort())
        # self.weights += torch.abs(self.weights[0]) + 1.0

    def get_gaussian_kernel(self, kernel_size):
        # self.weights *= self.variance[0].item()
        # weights = self.weights

        gaussian_dist = dist.Normal(0, 1)
        weights = gaussian_dist.sample((kernel_size,)).cuda()
        weights = torch.cat(weights.sort())
        weights += abs(weights[0].item())
        # weights[weights > 200] = 200
        # weights[weights < 0] = 0
        
        kernel_weights = torch.empty(kernel_size, kernel_size).cuda()
        for i in range(0, math.ceil(kernel_size / 2)):
            kernel_weights[i, i:kernel_size-i] = weights[i]
            kernel_weights[kernel_size - i - 1, i:kernel_size-i] = weights[i]
            for j in range(i + 1, kernel_size - i):
                kernel_weights[j, i] = weights[i]
                kernel_weights[j, kernel_size-i-1] = weights[i]
        # kernel_weights = kernel_weights.repeat
        # kernel_weights = torch.full((kernel_size, kernel_size), 1.0).cuda()
        # print(kernel_weights)
        # exit()
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