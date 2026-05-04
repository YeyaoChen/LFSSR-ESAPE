import torch
import torch.nn as nn
import torch.nn.functional as functional
from einops import rearrange
from model.module import *


class get_model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.channels = args.channels
        self.angRes = args.angRes

        # Stage 1-2: 2x, 4x
        self.lbp_enhancer = LBPEnhancer(self.channels, self.angRes, 4)

        scale_x2, disp_num_x2 = 2, 25
        scale_x4, disp_num_x4 = 1, 35
        self.plf_x2 = PLFConstruct(self.channels, scale_x2, self.angRes, 2.0, disp_num_x2)   # [(b ah aw),(s n c),h,w]
        self.plf_x4 = PLFConstruct(self.channels, scale_x4, self.angRes, 4.0, disp_num_x4)

        self.con_align = AlignModule(self.channels, self.angRes)
        self.fea_enhance = LFFEModule(self.channels, self.angRes, 4)
        self.lf_rec = RecConv(self.channels)

    def forward(self, in_lr, in_ref):
        # in_lr: [b,c,(ah h),(aw w)], in_ref: [b,c,αh,αw]

        #############################################  Stage1: 2x  #############################################
        in_b, in_c, _, _ = in_lr.shape
        in_lr = rearrange(in_lr, 'b c (ah h) (aw w) -> (b ah aw) c h w', ah=self.angRes, aw=self.angRes)    # [(b,ah,aw),1,h,w]
        lr_img_x2 = functional.interpolate(in_lr, scale_factor=2, mode="bicubic", align_corners=False)
        spa_lbp_x2, ang_lbp_x2, spa_lbp_img_x2, ang_lbp_img_x2 = self.lbp_enhancer(lr_img_x2)

        fea_x2 = self.con_align(lr_img_x2, self.plf_x2(in_ref))       # [(b,ah,aw),c,2h,2w]
        fea_x2 = self.fea_enhance(fea_x2, spa_lbp_x2, ang_lbp_x2)
        res_x2 = self.lf_rec(fea_x2)
        rec_x2 = res_x2 + lr_img_x2

        #############################################  Stage2: 4x  #############################################
        lr_img_x4 = functional.interpolate(rec_x2, scale_factor=2, mode="bicubic", align_corners=False)
        spa_lbp_x4, ang_lbp_x4, spa_lbp_img_x4, ang_lbp_img_x4 = self.lbp_enhancer(lr_img_x4)

        fea_x4 = self.con_align(lr_img_x4, self.plf_x4(in_ref))           # [(b,ah,aw),c,4h,4w]
        fea_x4 = self.fea_enhance(fea_x4, spa_lbp_x4, ang_lbp_x4)
        res_x4 = self.lf_rec(fea_x4)
        rec_x4 = res_x4 + lr_img_x4

        rec_lf = [to_sai_array(rec_x4, self.angRes), to_sai_array(rec_x2, self.angRes)]
        rec_spa_lbp = [to_sai_array(spa_lbp_img_x4, self.angRes), to_sai_array(spa_lbp_img_x2, self.angRes)]
        rec_ang_lbp = [ang_lbp_img_x4, ang_lbp_img_x2]

        return rec_lf, rec_spa_lbp, rec_ang_lbp


################################## Loss function ##################################
class get_loss(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.angRes = args.angRes
        self.spa_lbp = DifferentiableLBP(kernel_size=3, stride=1)
        self.ang_lbp = DifferentiableLBP(kernel_size=self.angRes, stride=self.angRes)
        self.l1 = nn.L1Loss()

    def forward(self, rec_x, spa_lbp, ang_lbp, gt):
        rec_x4, rec_x2 = rec_x[0], rec_x[1]
        spa_lbp_x4, spa_lbp_x2 = spa_lbp[0], spa_lbp[1]
        ang_lbp_x4, ang_lbp_x2 = ang_lbp[0], ang_lbp[1]
        gt_x4, gt_x2 = gt[0], gt[1]

        in_ah = self.angRes
        in_aw = self.angRes

        gt_spa_x4 = rearrange(gt_x4, 'b c (ah h) (aw w) -> (b ah aw) c h w', ah=in_ah, aw=in_aw)
        gt_spa_x2 = rearrange(gt_x2, 'b c (ah h) (aw w) -> (b ah aw) c h w', ah=in_ah, aw=in_aw)
        gt_spa_x4 = to_sai_array(self.spa_lbp(gt_spa_x4), self.angRes)
        gt_spa_x2 = to_sai_array(self.spa_lbp(gt_spa_x2), self.angRes)

        gt_ang_x4 = self.ang_lbp(rearrange(gt_x4, 'b c (ah h) (aw w) -> b c (h ah) (w aw)', ah=self.angRes, aw=self.angRes))
        gt_ang_x2 = self.ang_lbp(rearrange(gt_x2, 'b c (ah h) (aw w) -> b c (h ah) (w aw)', ah=self.angRes, aw=self.angRes))

        rec_loss = self.l1(rec_x4, gt_x4) + self.l1(rec_x2, gt_x2)
        spa_lbp_loss = self.l1(spa_lbp_x4, gt_spa_x4) + self.l1(spa_lbp_x2, gt_spa_x2)
        ang_lbp_loss = self.l1(ang_lbp_x4, gt_ang_x4) + self.l1(ang_lbp_x2, gt_ang_x2)

        total_loss = rec_loss + spa_lbp_loss * 0.1 + ang_lbp_loss * 0.1
        return total_loss


def weights_init(m):
    pass


def to_sai_array(x, ang_res):
    x = rearrange(x, '(b ah aw) c h w -> b c (ah h) (aw w)', ah=ang_res, aw=ang_res)
    return x


if __name__ == '__main__':
    from thop import profile
    import argparse
    parser = argparse.ArgumentParser(description="Light field image super-resolution -- train mode")
    parser.add_argument("--angRes", type=int, default=5, help="Angular resolution of light field")
    parser.add_argument("--channels", type=int, default=48, help="Number of channels")
    args = parser.parse_args()

    model = get_model(args).cuda()
    lr = torch.randn(1, 1, 5*24, 5*24).cuda()       # [b,1,ah*h,aw*w]
    gt1 = torch.randn(1, 1, 5*96, 5*96).cuda()      # [b,1,ah*h,aw*w]
    gt2 = torch.randn(1, 1, 5*48, 5*48).cuda()      # [b,1,ah*h,aw*w]
    gt = [gt1, gt2]
    ref = torch.randn(1, 1, 96, 96).cuda()          # [b,1,H_ref,W_ref]
    # out_x, spa_lbp, ang_lbp = model(lr, ref)

    # print(out_x[0].shape, out_x[1].shape, out_x[2].shape)
    # print(spa_lbp[0].shape, spa_lbp[1].shape, spa_lbp[2].shape)
    # print(ang_lbp[0].shape, ang_lbp[1].shape, ang_lbp[2].shape)

    # loss = get_loss(args).cuda()
    # x = loss(out_x, spa_lbp, ang_lbp, gt)
    # print(x)

    total = sum([param.nelement() for param in model.parameters()])
    flops, params = profile(model, inputs=(lr, ref,))
    print('Number of parameters: %.2fM' % (total / 1e6))
    print('Number of FLOPs: %.2fG' % (flops / 1e9))