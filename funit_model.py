"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license
(https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import copy

import torch
import torch.nn as nn

from model.FUNIT.networks import FewShotGen, GPPatchMcResDis
from model.FUNIT.utils import get_dichomy_loader, kl_divergence, sim

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
            height=hp['crop_image_height'],
            width=hp['crop_image_width'],
            crop=True,
            num_workers=1,
            n_cls=hp['pool_size'])
        self.step = 0


    def forward(self, co_data, cl_data, cn_data, hp, mode):
        xa = co_data[0].cuda()
        la = co_data[1].cuda()
        xb = cl_data[0].cuda()
        lb = cl_data[1].cuda()
        xn = cn_data.cuda()
        if mode == 'gen_update':
            # blur_xa = self.gen.erase(xa)
            # blur_xb = self.gen.erase(xb)
            # c_xa = self.gen.enc_content(blur_xa)
            # c_xa = self.gen.enc_content(xa)
            # c_xb = self.gen.enc_content(xb)
            # c_xn = self.gen.enc_content(xn)
            # s_xa = self.gen.enc_class_model(xa)
            # s_xb = self.gen.enc_class_model(xb)
            # s_xn = self.gen.enc_class_model(xn)
            # c_xa_feature = c_xa.flatten(start_dim=2, end_dim=-1)
            # c_xb_feature = c_xb
            # combined_features = 1 * c_xa_feature
            translated_out = self.gen.affine(xa)
            # translated_out = translated_out.reshape(c_xa.shape)

            # xt = self.gen.decode(translated_out, s_xa) # generation
            # # xr = self.gen.decode(clear_xa, s_xa)  # pose reconstruction
            # xb_r = self.gen.decode(c_xb, s_xb)
            # xn_r = self.gen.decode(c_xn, s_xn)
            # # proto_sim_set = sim(c_xa.mean((2,3)), prototype_emb.mean((1,2)))# batch, emb x 1, emb -> batch, 1
            # # print(proto_sim_set)
            # # exit()
            # # max, max_i = torch.max(proto_sim_set)
            # # xp2 = xa[max_i[0].item(), :, :, :].repeat(len(c_xn), 1, 1, 1)
            # l_adv_t, class_loss,\
            #       gacc_t, xt_gan_feat = self.dis.calc_gen_loss(xt, lb, \
            #     counterpart=xb_r.detach(), original=None, challenge=xn_r.detach())
            # # sim_r, l_adv_r, gacc_r, xr_gan_feat = self.dis.calc_gen_loss(xr, la, counterpart=xr.detach(), original=xb.detach(), challenge=xn.detach())
            # l_contrast = self.dis.calc_contrast_loss(xt, lb) * 0.1


            # idxs = torch.randperm(len(c_xb)).cuda()
            # counterpart = torch.index_select(c_xb, 0, idxs)
            # class_loss = -torch.log(sim(translated_out.mean((2,3)).detach(), \
            #                              counterpart.mean((2,3)).detach())).mean()
            # class_loss2 = sim(translated_out.mean((2,3)).detach(), \
            #                              c_xn.mean((2,3)).detach()).mean() * 0.5
            # # print(l_contrast.shape)
            # # _, _, xb_gan_feat = self.dis(xb)
            # # _, _, xa_gan_feat = self.dis(xa)
            # prototype_emb = c_xb_feature.mean(0).repeat(32, 1, 1, 1)# 512, 16, 16
            # # print(translated_out.shape, prototype_emb.shape)
            # exit()
            # kl_loss = kl_divergence(translated_out.mean((2,3)),
            #                           prototype_emb.mean((2,3))) * 500
            # l_sim = -torch.log(sim(translated_out, prototype_emb) + 1e-5)
            # l_sim = l_sim.mean()
            # l_m_rec = recon_criterion(xt_gan_feat,
            #                           xb_gan_feat)
            # l_c_rec = recon_criterion(c_xa,
            #                           xa_ground_truth)
            # l_m_rec = recon_criterion(c_xa,
            #                           xb_ground_truth)
            # l_x_rec = (recon_criterion(xn_r, xn) + recon_criterion(xb_r, xb)) * 1
            # l_x_rec += recon_criterion(xt, xa) * 1
            # acc = gacc_t
            # # l_contrast *= 1
            # # l_adv_t *= 1
            
            # if self.step % 200 == 0:
            #     print('recon', l_x_rec, 'kl_loss', kl_loss,\
            #           'class_l', class_loss, 'contrast_l', l_contrast, 'class_loss2', class_loss2)
            # self.step += 1
            # l_total = l_x_rec + kl_loss + l_adv_t + class_loss + l_contrast + class_loss2

            l_total = recon_criterion(translated_out, xb.mean(0))
            l_total += recon_criterion(translated_out, xa) * 0.5
            if self.step % 20 == 0:
                print(l_total)
            # l_total = (hp['gan_w'] * (class_loss + target_l + l_x_rec)+ hp[
            #     'gan_w'] * (l_contrast + l_adv_t))
            # l_total = (hp['gan_w'] * (l_contrast + l_adv_t + class_loss + l_x_rec))
            l_total.backward()
            return l_total, l_total, l_total, l_total, l_total, l_total
            # return l_total, l_x_rec, kl_loss, class_loss, l_adv_t, acc ## slack
        elif mode == 'dis_update':
            xb.requires_grad_()
            l_real_pre, acc_r, resp_r = self.dis.calc_dis_real_loss(xb)
            l_real = hp['gan_w'] * l_real_pre
            # l_real *= 1
            l_real.backward(retain_graph=True)
            l_reg_pre = self.dis.calc_grad2(resp_r, xb)
            l_reg = 1 * l_reg_pre # 10 *
            l_reg.backward()
            with torch.no_grad():
                xa = self.gen.erase(xa)
                c_xa = self.gen.enc_content(xa)
                c_xb = self.gen.enc_content(xb)
                c_xa = 1 * c_xa + (1 - 1) * c_xb
                s_xa = self.gen.enc_class_model(xa)
                translated_out = c_xa.flatten(start_dim=2, end_dim=-1)
                translated_out = self.gen.affine(translated_out)
                translated_out = translated_out.reshape(c_xa.shape)
                xt = self.gen.decode(translated_out, s_xa)
            l_fake_p, acc_f, resp_f = self.dis.calc_dis_fake_loss(xt.detach())
            l_fake = hp['gan_w'] * l_fake_p
            if self.step % 20 == 0:
                print(l_fake, 'fake', l_real, 'real')
            l_fake.backward()
            l_total = l_fake + l_real + l_reg
            acc = 1 * (acc_f + acc_r)
            return l_total, l_fake_p, l_real_pre, l_reg_pre, acc
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
        # c_xa = self.gen.enc_content(xa)
        # c_xb = self.gen.enc_content(xb)
        # c_xa_current = 1 * c_xa + (1 - 1) * c_xb
        # s_xa_current = self.gen.enc_class_model(xa)
        # s_xb_current = self.gen.enc_class_model(xb)
        # translated_out = c_xa_current.flatten(start_dim=2, end_dim=-1)
        # translated_out = self.gen.affine(translated_out)
        # translated_out = translated_out.reshape(c_xa_current.shape)
        # xt_current = self.gen.decode(translated_out, s_xb_current)
        # xr_current = self.gen.decode(c_xa, s_xa_current)
        translated_out = self.gen.affine(xa)
        # c_xa = self.gen_test.enc_content(xa)
        # s_xa = self.gen_test.enc_class_model(xa)
        # translated_out = c_xa.flatten(start_dim=2, end_dim=-1)
        # translated_out = self.gen_test.affine(translated_out)
        # translated_out = translated_out.reshape(c_xa_current.shape)
        # xt = self.gen_test.decode(translated_out, s_xa)
        # xr = self.gen_test.decode(c_xa, s_xa)
        self.train()
        return xa, translated_out, xb#, xr_current, xt_current

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
    
    # optionally returns qry expansions of size: (expansion_size + 1, 3, h, w)
    # 'pool_size' copies of candidate neighbours are randomly sampled
    # translations are conducted only with the best 'expansion_size' candidates
    # best candidates are defined as those with the highest vector dot product
    # the vectors are features learnt by picker
    def pick_traffic(self, qry, expansion_size=0, get_img = False, random=False): # only one qry
        # pool size should be <= class numbers ##slack
        candidate_neighbours = next(iter(self.train_loader)) # from train sampler, size: pool_size, 3, h, w
        candidate_neighbours = candidate_neighbours[0].cuda()
        _, _, qry_features = self.dis(qry) # batch=1, feature_size
        _, _, nb_features = self.dis(candidate_neighbours)
        with torch.no_grad():
            nb_features_trans = nb_features.transpose(1,0)
            scores = torch.mm(qry_features, nb_features_trans) # q qries, n neighbors
        if random == False:
            scores, idxs = torch.sort(scores)
            selected_nbs = candidate_neighbours[idxs][:expansion_size, :, :, :]
        else:
            selected_nbs = candidate_neighbours[:expansion_size, :, :, :]
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
            for selected_i in range(expansion_size + 1):
                translation = translations[selected_i]
                image = translation.detach().cpu().squeeze().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = ((image + 1) * 1 * 255.0)
                output_img = Image.fromarray(np.uint8(image))
                output_img.save(f'./images/output{selected_i}.jpg', 'JPEG', quality=99)
                print('Save output')
            print(torch.stack(translations).shape)
        if get_img == False:
            return torch.stack(translations).squeeze()
    
    def pick_animals(self, qry, expansion_size=0, get_img = False, random=False, img_id=None): # only one qry
        # pool size should be <= class numbers ##slack
        candidate_neighbours = next(iter(self.train_loader)) # from train sampler, size: pool_size, 3, h, w
        candidate_neighbours = candidate_neighbours[0].cuda()
        with torch.no_grad():
            qry_features = self.gen.enc_content(qry).mean((2,3)) # batch=1, feature_size
            nb_features = self.gen.enc_content(candidate_neighbours).mean((2,3))
            nb_features_trans = nb_features.transpose(1,0)
            scores = torch.mm(qry_features, nb_features_trans).squeeze() # q qries, n neighbors
        if random == False:
            scores, idxs = torch.sort(scores, descending=True) # more similar in front
            idxs = idxs.long()
            selected_nbs = candidate_neighbours.index_select(dim=0, index=idxs)
            selected_nbs = selected_nbs[:expansion_size, :, :, :]
        else:
            selected_nbs = candidate_neighbours[:expansion_size, :, :, :]
        class_code = self.compute_k_style(qry, 1)
        translations = [self.translate_simple(qry, class_code)]
        with torch.no_grad():
            for selected_i in range(expansion_size):
                nb = selected_nbs[selected_i, :, :, :].unsqueeze(0)
                translation = self.translate_simple(nb, class_code)
                translations.append(translation)
        if get_img == True:
            import numpy as np
            from PIL import Image
            for selected_i in range(expansion_size + 1):
                translation = translations[selected_i]
                image = translation.detach().cpu().squeeze().numpy()
                image = np.transpose(image, (1, 2, 0))
                image = ((image + 1) * 1 * 255.0)
                output_img = Image.fromarray(np.uint8(image))
                output_img.save(\
                    f'/home/nus/Documents/research/augment/code/FEAT/model/FUNIT/images/output{img_id}_{selected_i}.jpg', 'JPEG', quality=99)
                print('Save output')
        return torch.stack(translations).squeeze()

# yaml should contain original encoder path, and set poolsize and other hp