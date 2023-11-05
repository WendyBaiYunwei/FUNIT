"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn

from networks import FewShotGen, GPPatchMcResDis
from utils import get_dichomy_loader

def recon_criterion(predict, target):
    return torch.mean(torch.abs(predict - target))


class FUNITModel(nn.Module):
    def __init__(self, hp):
        super(FUNITModel, self).__init__()
        self.gen = FewShotGen(hp['gen'])
        self.dis = GPPatchMcResDis(hp['dis'])
        self.gen_test = copy.deepcopy(self.gen)
        self.train_loader = get_dichomy_loader(
            episodes=hp['max_iter'],
            root=hp['data_folder_train'],
            file_list=hp['data_list_train'],
            batch_size=1,
            new_size=hp['new_size'],
            height=hp['height'],
            width=hp['width'],
            crop=True,
            num_workers=1,
            n_cls=hp['pool_size'])

    def forward(self, co_data, cl_data, cn_data, hp, mode):
        xa = co_data[0].cuda()
        la = co_data[1].cuda()
        xb = cl_data[0].cuda()
        lb = cl_data[1].cuda()
        xn = cn_data.cuda()
        if mode == 'gen_update':
            c_xa = self.gen.enc_content(xa)
            s_xa = self.gen.enc_class_model(xa)
            s_xb = self.gen.enc_class_model(xb)
            xt = self.gen.decode(c_xa, s_xb)  # translation
            xr = self.gen.decode(c_xa, s_xa)  # reconstruction
            l_adv_t, gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb, xb.detach(), xa.detach(), xn.detach())
            l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la, xa.detach(), xb.detach(), xn.detach())
            _, _, xb_gan_feat = self.dis(xb)
            _, _, xa_gan_feat = self.dis(xa)
            l_c_rec = recon_criterion(xr_gan_feat,
                                      xa_gan_feat)
            l_m_rec = recon_criterion(xt_gan_feat,
                                      xb_gan_feat)
            l_x_rec = recon_criterion(xr, xa)
            l_adv = 0.5 * (l_adv_t + l_adv_r)
            acc = 0.5 * (gacc_t + gacc_r)
            l_total = (hp['gan_w'] * l_adv + hp['r_w'] * l_x_rec + hp[
                'fm_w'] * (l_c_rec + l_m_rec))
            l_total.backward()
            return l_total, l_adv, l_x_rec, l_c_rec, l_m_rec, acc
        elif mode == 'dis_update':
            xb.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb)
            l_real = hp['gan_w'] * l_real_pre
            l_real.backward(retain_graph=True)
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 10 * l_reg_pre
            l_reg.backward()
            with torch.no_grad():
                c_xa = self.gen.enc_content(xa)
                s_xb = self.gen.enc_class_model(xb)
                xt = self.gen.decode(c_xa, s_xb)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach())
            l_fake = hp['gan_w'] * l_fake_p
            l_fake.backward()
            l_total = l_fake + l_real #+loss_reg
            acc = 0.5 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, torch.zeros(l_real_pre.shape), acc
        elif mode == 'picker_update':
            _, _, qry_features = self.dis(cl_data) # batch, q, feature_size
            _, _, nb_features = self.dis(co_data) # qries and nbs are of different classes
            matrix_forward = torch.bmm(qry_features, nb_features.transpose(2,1)) # q qries, n neighbors
            matrix_reverse = torch.bmm(nb_features, qry_features.transpose(2,1))
            scores_forward = self.get_score(qry = cl_data, nb = co_data, cn_data = xn)
            scores_reverse = self.get_score(qry = co_data, nb = cl_data, cn_data = xn)
            loss_forward = recon_criterion(matrix_forward, scores_forward)
            loss_reverse = recon_criterion(matrix_reverse, scores_reverse)
            loss = loss_forward + loss_reverse
            loss.backward()
            return loss
        else:
            assert 0, 'Not support operation'

    def test(self, co_data, cl_data):
        self.eval()
        self.gen.eval()
        self.gen_test.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen.enc_content(xa)
        s_xa_current = self.gen.enc_class_model(xa)
        s_xb_current = self.gen.enc_class_model(xb)
        xt_current = self.gen.decode(c_xa_current, s_xb_current)
        xr_current = self.gen.decode(c_xa_current, s_xa_current)
        c_xa = self.gen_test.enc_content(xa)
        s_xa = self.gen_test.enc_class_model(xa)
        s_xb = self.gen_test.enc_class_model(xb)
        xt = self.gen_test.decode(c_xa, s_xb)
        xr = self.gen_test.decode(c_xa, s_xa)
        self.train()
        return xa, xr_current, xt_current, xb, xr, xt

    def translate_k_shot(self, co_data, cl_data, k):
        self.eval()
        xa = co_data[0].cuda()
        xb = cl_data[0].cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        if k == 1:
            s_xb_current = self.gen_test.enc_class_model(xb)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        else:
            s_xb_current_before = self.gen_test.enc_class_model(xb)
            s_xb_current_after = s_xb_current_before.squeeze(-1).permute(1,
                                                                         2,
                                                                         0)
            s_xb_current_pool = torch.nn.functional.avg_pool1d(
                s_xb_current_after, k)
            s_xb_current = s_xb_current_pool.permute(2, 0, 1).unsqueeze(-1)
            xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def compute_k_style(self, style_batch, k):
        self.eval()
        style_batch = style_batch.cuda()
        s_xb_before = self.gen_test.enc_class_model(style_batch)
        s_xb_after = s_xb_before.squeeze(-1).permute(1, 2, 0)
        s_xb_pool = torch.nn.functional.avg_pool1d(s_xb_after, k)
        s_xb = s_xb_pool.permute(2, 0, 1).unsqueeze(-1)
        return s_xb

    def translate_simple(self, content_image, class_code):
        self.eval()
        xa = content_image.cuda()
        s_xb_current = class_code.cuda()
        c_xa_current = self.gen_test.enc_content(xa)
        xt_current = self.gen_test.decode(c_xa_current, s_xb_current)
        return xt_current

    def get_score(self, qry, nb, cn_data):
        with torch.no_grad():
            c_xa = self.gen.enc_content(nb.detach())
            # s_xa = self.gen.enc_class_model(xa)
            s_xb = self.gen.enc_class_model(qry.detach())
            translation = self.gen.decode(c_xa, s_xb)
        real_degree = self.dis(translation, qry, nb, cn_data, selector=True)# how real the generation appears
        # and is the generation similar to the right class?
        return real_degree
    
    # optionally returns qry expansions of size: (expansion_size, 3, h, w)
    # 'pool_size' copies of candidate neighbours are randomly sampled
    # translations are conducted only with the best 'expansion_size' candidates
    # best candidates are defined as those with the highest vector dot product
    # the vectors are features learnt by picker
    def pick(self, qry, expansion_size, get_img = False): # only one qry
        # pool size should be <= class numbers ##slack
        expansion_size += 1
        candidate_neighbours = next(iter(self.train_loader)) # from train sampler, size: pool_size, 3, h, w
        candidate_neighbours = candidate_neighbours[0].cuda()
        _, _, qry_features = self.dis(qry) # batch=1, feature_size
        _, _, nb_features = self.dis(candidate_neighbours)
        scores = []
        with torch.no_grad():
            scores = torch.mm(qry_features, nb_features.transpose(1,0)) # q qries, n neighbors
            print(scores.shape)
            exit()
        scores, idxs = torch.sort(torch.stack(scores))
        selected_nbs = candidate_neighbours[idxs][:expansion_size, :, :, :]
        class_code = self.compute_k_style(qry, 1)
        translations = [self.translate_simple(qry, class_code)]
        with torch.no_grad():
            for selected_i in range(expansion_size):
                nb = selected_nbs[selected_i, :, :, :]
                translation = self.translate_simple(nb, class_code)
                translations.append(translation)
        if get_img == True:
            import numpy as np
            from PIL import Image
            for selected_i in range(expansion_size):
                translation = translations[selected_i]
                image = translation.detach().cpu().squeeze().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = ((image + 1) * 0.5 * 255.0)
                output_img = Image.fromarray(np.uint8(image))
                output_img.save(f'./output/images/output{selected_i}', 'JPEG', quality=99)
                print('Save output')
        if get_img == False:
            return torch.stack(translations)

# yaml should contain original encoder path, and set poolsize and other hp