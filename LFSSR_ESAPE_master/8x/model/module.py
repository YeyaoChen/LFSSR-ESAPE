import torch
import torch.nn as nn
import torch.nn.functional as functional
import numpy as np
from einops import rearrange
from utils.softsplat import FunctionSoftsplat


##########################################  Main module  ##########################################
class LBPEnhancer(nn.Module):
    def __init__(self, channels, ang_res, lbp_num):
        super(LBPEnhancer, self).__init__()
        self.ang_res = ang_res
        self.spa_lbp = DifferentiableLBP(kernel_size=3, stride=1)
        self.ang_lbp = DifferentiableLBP(kernel_size=ang_res, stride=ang_res)
        self.init_spa_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.init_ang_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        spa_blocks = [ResCABlock(channels) for _ in range(lbp_num)]
        self.sap_lbp_enhancer = nn.Sequential(*spa_blocks)
        ang_blocks = [ResCABlock(channels) for _ in range(lbp_num)]
        self.ang_lbp_enhancer = nn.Sequential(*ang_blocks)

        self.tail_spa_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))
        self.tail_ang_conv = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True))

        self.out_spa_conv = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)
        self.out_ang_conv = nn.Conv2d(in_channels=channels, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        _, _, in_h, in_w = in_x.shape
        in_spa_lbp = self.spa_lbp(in_x)
        spa_x = self.init_spa_conv(in_spa_lbp)
        spa_x = self.sap_lbp_enhancer(spa_x)
        spa_x = self.tail_spa_conv(spa_x)
        out_spa_lbp = self.out_spa_conv(spa_x) + in_spa_lbp

        # in_ang_lbp =self.ang_lbp(rearrange(in_x, '(b ah aw) c h w -> (b h w) c ah aw', ah=self.ang_res, aw=self.ang_res))
        # in_ang_lbp = rearrange(in_ang_lbp.squeeze(dim=-1).squeeze(dim=-1), '(b h w) c -> b c h w', h=in_h, w=in_w)
        in_ang_lbp = self.ang_lbp(rearrange(in_x, '(b ah aw) c h w -> b c (h ah) (w aw)', ah=self.ang_res, aw=self.ang_res))
        ang_x = self.init_ang_conv(in_ang_lbp)
        ang_x = self.ang_lbp_enhancer(ang_x)
        ang_x = self.tail_ang_conv(ang_x)
        out_ang_lbp = self.out_ang_conv(ang_x) + in_ang_lbp
        return spa_x, ang_x, out_spa_lbp, out_ang_lbp


class PLFConstruct(nn.Module):
    def __init__(self, channels, scale, ang_res, disp_range, disp_num):
        super(PLFConstruct, self).__init__()
        self.scale = scale ** 2
        self.ang_res = ang_res
        self.disp_range = disp_range
        self.disp_num = disp_num
        self.pus = nn.PixelUnshuffle(scale)
        self.conv_dd = nn.Conv2d(scale**2*disp_num, channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, in_x):
        # [b,c,h,w]
        in_b, in_c, _, _ = in_x.shape
        out_x = self.pus(in_x)           # [b,s^2c,h/s,w/s]
        out_x = rearrange(out_x, 'b (s c) h w -> (b s) c h w', s=self.scale, c=in_c)
        out_x = center_to_boundary(out_x, self.ang_res, self.disp_range, self.disp_num)     # [b,n,c,ah,aw,h,w]
        out_x = rearrange(out_x, '(b s) n c ah aw h w -> (b ah aw) (s n c) h w', b=in_b, s=self.scale)
        out_x = self.conv_dd(out_x)
        return out_x


class AlignModule(nn.Module):
    def __init__(self, channels, ang_res):
        super(AlignModule, self).__init__()
        self.ang_res = ang_res
        self.lf_sfe = LRLFShallFE(channels)
        self.plf_sfe = PLFShallFE(channels)
        self.conv_q = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv1x1 = nn.Conv2d(channels*3, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_lr, hr_plf):
        # [(b,ah,aw),c,h,w] & [(b,ah,aw),nc,h,w]
        lr_fea = self.lf_sfe(in_lr)
        hr_fea = self.plf_sfe(hr_plf)
        fea_q = self.conv_q(lr_fea)
        fea_k = self.conv_k(hr_fea)

        fea_q = rearrange(fea_q, '(b ah aw) c h w -> b c (ah aw h w)', ah=self.ang_res, aw=self.ang_res)
        fea_k = rearrange(fea_k, '(b ah aw) c h w -> b c (ah aw h w)', ah=self.ang_res, aw=self.ang_res)
        fea_v = rearrange(hr_fea, '(b ah aw) c h w -> b c (ah aw h w)', ah=self.ang_res, aw=self.ang_res)

        fea_q = functional.normalize(fea_q, dim=-1)
        fea_k = functional.normalize(fea_k, dim=-1)

        att_map = (fea_q @ fea_k.transpose(-2, -1))
        att_map = att_map.softmax(dim=-1)

        align_fea = (att_map @ fea_v)
        align_fea = rearrange(align_fea, 'b c (ah aw h w) -> (b ah aw) c h w', ah=self.ang_res, aw=self.ang_res, h=lr_fea.shape[-2], w=lr_fea.shape[-1])
        cat_fea = torch.cat((lr_fea, hr_fea, align_fea), dim=1)

        out_fea = self.conv1x1(cat_fea)
        out_fea = self.conv3x3(self.act(out_fea))
        return out_fea + lr_fea


class LFFEModule(nn.Module):
    def __init__(self, channels, ang_res, lf_num):
        super(LFFEModule, self).__init__()
        self.lfe_module = nn.ModuleList([LFFEGroup(channels, ang_res) for _ in range(lf_num)])

    def forward(self, in_x, in_spa_lbp, in_ang_lbp):
        # [(b,ah,aw),c,h,w]
        out_x = in_x
        for lfe_ind in self.lfe_module:
            out_x = lfe_ind(out_x, in_spa_lbp, in_ang_lbp)
        return out_x


class RecConv(nn.Module):
    def __init__(self, channels):
        super(RecConv, self).__init__()
        self.rec_conv = nn.Conv2d(channels, 1, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        out_x = self.rec_conv(in_x)
        return out_x


class UpSample(nn.Module):
    def __init__(self, channels, factor):
        super(UpSample, self).__init__()
        self.up_conv = nn.Sequential(
            nn.Conv2d(channels, channels * factor ** 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.PixelShuffle(factor))

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        out_x = self.up_conv(in_x)
        return out_x


##########################################  Sub module  ##########################################
class PLFShallFE(nn.Module):
    def __init__(self, channels):
        super(PLFShallFE, self).__init__()
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv4 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        buffer_x = self.act( self.conv_in(in_x))

        res_x1 = self.act(self.res_conv1(buffer_x))
        out_x1 = self.res_conv2(res_x1) + buffer_x

        res_x2 = self.act(self.res_conv3(out_x1))
        out_x = self.res_conv4(res_x2) + out_x1
        return out_x


class LRLFShallFE(nn.Module):
    def __init__(self, channels):
        super(LRLFShallFE, self).__init__()
        self.conv_in = nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        buffer_x = self.act(self.conv_in(in_x))

        res_x = self.act(self.res_conv1(buffer_x))
        out_x = self.res_conv2(res_x) + buffer_x
        return out_x


class LFFEGroup(nn.Module):
    def __init__(self, channels, ang_res):
        super(LFFEGroup, self).__init__()
        self.lf_conv = nn.ModuleList([LFConv(channels, ang_res) for _ in range(2)])
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x, in_spa_lbp, in_ang_lbp):
        # [(b,ah,aw),c,h,w]
        out_x = in_x
        for conv_ind in self.lf_conv:
            out_x = conv_ind(out_x, in_spa_lbp, in_ang_lbp)

        out_x = self.conv(self.act(out_x))
        return out_x + in_x


class LFConv(nn.Module):
    def __init__(self, channels, ang_res):
        super(LFConv, self).__init__()
        self.ang_res = ang_res
        self.conv1 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv2 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.conv3x3_1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3x3_2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.sav_conv = SAVConv(channels, ang_res)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x, in_spa_lbp, in_ang_lbp):
        # [(b,ah,aw),c,h,w]
        spa_x = self.conv1(torch.cat((in_x, in_spa_lbp), dim=1))
        spa_x = self.conv3x3_1(self.act(spa_x)) + in_x

        in_ang_lbp = in_ang_lbp.repeat(self.ang_res*self.ang_res, 1, 1, 1)
        ang_x = self.conv2(torch.cat((in_x, in_ang_lbp), dim=1))
        ang_x = self.conv3x3_2(self.act(ang_x)) + in_x

        out_x = self.conv3(torch.cat((spa_x, ang_x), dim=1))
        out_x = self.sav_conv(out_x)
        return out_x


class SASConv(nn.Module):
    def __init__(self, channels, ang_res):
        super(SASConv, self).__init__()
        self.ang_res = ang_res
        self.spa_conv = FreqConv(channels)
        self.ang_conv = FreqConv(channels)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        spa_x = self.spa_conv(in_x)
        buffer_x = rearrange(spa_x, '(b ah aw) c h w -> (b h w) c ah aw', ah=self.ang_res, aw=self.ang_res)

        ang_x = self.ang_conv(buffer_x)
        out_x = rearrange(ang_x, '(b h w) c ah aw -> (b ah aw) c h w', h=in_x.shape[-2], w=in_x.shape[-1])
        return out_x


class SACConv(nn.Module):
    def __init__(self, channels, ang_res):
        super(SACConv, self).__init__()
        self.ang_res = ang_res
        self.hepi_conv = FreqConv(channels)
        self.vepi_conv = FreqConv(channels)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        hepi_x = rearrange(in_x, '(b ah aw) c h w -> (b ah h) c aw w', ah=self.ang_res, aw=self.ang_res)
        hepi_x = self.hepi_conv(hepi_x)

        vepi_x = rearrange(hepi_x, '(b ah h) c aw w -> (b aw w) c ah h', ah=self.ang_res, h=in_x.shape[-2])
        vepi_x = self.vepi_conv(vepi_x)
        out_x = rearrange(vepi_x, '(b aw w) c ah h -> (b ah aw) c h w', aw=self.ang_res, w=in_x.shape[-1])
        return out_x


class SAVConv(nn.Module):
    def __init__(self, channels, ang_res):
        super(SAVConv, self).__init__()
        self.sas_conv = SASConv(channels, ang_res)
        self.sac_conv = SACConv(channels, ang_res)
        self.conv_agg = nn.Conv2d(channels*2, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, in_x):
        # [(b,ah,aw),c,h,w]
        out_x1 = self.sas_conv(in_x)
        out_x2 = self.sac_conv(in_x)
        out_x = self.conv_agg(torch.cat((out_x1, out_x2), dim=1))
        return out_x + in_x


class FreqConv(nn.Module):
    def __init__(self, channels):
        super(FreqConv, self).__init__()
        self.conv3x3 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.freq_conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(channels * 2, channels * 2, kernel_size=1, stride=1, padding=0, bias=False))
        self.dw_conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, groups=channels, bias=False)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [b,c,h,w]
        in_x = self.act(self.conv3x3(in_x))
        x_fft = torch.fft.fft2(in_x, norm='backward')
        x_fft = torch.cat((x_fft.real, x_fft.imag), dim=1)
        x_fft =  self.freq_conv(x_fft)
        x_real, x_imag = torch.chunk(x_fft, 2, dim=1)
        fft_result = torch.complex(x_real, x_imag)
        out_x = torch.fft.ifft2(fft_result, dim=(-2, -1), norm='backward')
        out_x = self.act(self.dw_conv(torch.abs(out_x)))
        return out_x


class ResCABlock(nn.Module):
    def __init__(self, channels):
        super(ResCABlock, self).__init__()
        self.res_conv1 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.res_conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.ca_layer = CALayer(channels, reduction=8)
        self.act = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, in_x):
        # [b,c,h,w]
        res_x = self.act(self.res_conv1(in_x))
        res_x = self.res_conv2(res_x)
        res_x = self.ca_layer(res_x)
        out_x = res_x + in_x
        return out_x


class CALayer(nn.Module):
    def __init__(self, channels, reduction=8):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels=channels, out_channels=channels//reduction, kernel_size=1, stride=1, padding=0, bias=False),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Conv2d(in_channels=channels//reduction, out_channels=channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.Sigmoid())

    def forward(self, in_x):
        gap_x = self.avg_pool(in_x)
        attn_x = self.conv_du(gap_x)
        out_x = attn_x * in_x
        return out_x


class DifferentiableLBP(nn.Module):
    def __init__(self, kernel_size=3, stride=1, alpha=10.0):
        super(DifferentiableLBP, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.alpha = alpha

        num_neighbors = kernel_size ** 2 - 1
        self.register_buffer('coff', torch.tensor(2.0 ** num_neighbors - 1))

        power_weights = 2 ** torch.arange(num_neighbors).float()
        self.register_buffer('power_weights', power_weights.view(1, num_neighbors, 1, 1))

        neighbor_weight = torch.zeros((num_neighbors, 1, kernel_size, kernel_size))
        center_weight = torch.zeros((1, 1, kernel_size, kernel_size))

        center_idx = kernel_size // 2
        center_weight[0, 0, center_idx, center_idx] = 1.0

        n = 0
        for i in range(kernel_size):
            for j in range(kernel_size):
                if i == center_idx and j == center_idx:
                    continue
                neighbor_weight[n, 0, i, j] = 1.0
                n += 1

        self.register_buffer('neighbor_filter', neighbor_weight)
        self.register_buffer('center_filter', center_weight)

    def forward(self, in_x):
        b, c, h, w = in_x.shape
        padding = self.k // 2
        dtype = in_x.dtype

        n_filter = self.neighbor_filter.to(dtype).repeat(c, 1, 1, 1)
        c_filter = self.center_filter.to(dtype).repeat(c, 1, 1, 1)

        out_neighbors = functional.conv2d(in_x, n_filter, padding=padding, stride=self.stride, groups=c)
        center_pixels = functional.conv2d(in_x, c_filter, padding=padding, stride=self.stride, groups=c)

        out_neighbors = out_neighbors.view(b, c, -1, out_neighbors.shape[-2], out_neighbors.shape[-1])
        center_pixels = center_pixels.unsqueeze(2)

        diff = out_neighbors - center_pixels
        binary_codes = torch.sigmoid(self.alpha * diff)

        lbp_result = torch.sum(binary_codes * self.power_weights.to(dtype), dim=2)
        return lbp_result / self.coff.to(dtype)


def center_to_boundary(central_img, ang_res, disp_range, disp_num):
    # [b,c,h,w]
    in_b, in_c, in_h, in_w = central_img.shape
    device = central_img.device
    disp_values = generate_disparity_value(disp_range, disp_num//2).to(device).type_as(central_img)

    central_x = ang_res // 2
    central_y = ang_res // 2
    ay, ax = torch.meshgrid(torch.arange(ang_res, device=device), torch.arange(ang_res, device=device), indexing='ij')

    diff_x = (central_x - ax).float()
    diff_y = (central_y - ay).float()

    flow_x = disp_values[:, None, None] * diff_x[None, :, :]  # [disp_num, ah, aw]
    flow_y = disp_values[:, None, None] * diff_y[None, :, :]  # [disp_num, ah, aw]
    flows = torch.stack([flow_x, flow_y], dim=3)  # [disp_num, ah, aw, 2]
    flows = rearrange(flows, 'n ah aw d -> (n ah aw) d 1 1')
    full_grid = flows.expand(-1, -1, in_h, in_w)  # [Total_Batch, 2, h, w]

    input_repeated = central_img.repeat_interleave(disp_num * ang_res * ang_res, dim=0)
    warp_all = FunctionSoftsplat(tenInput=input_repeated, tenFlow=full_grid.repeat(in_b, 1, 1, 1), tenMetric=None, strType='average')

    psv_lf = rearrange(warp_all, '(b n ah aw) c h w -> b n c ah aw h w', b=in_b, n=disp_num, ah=ang_res, aw=ang_res)
    return psv_lf


def generate_disparity_value(disp_range, half_disp_num):
    indices = torch.arange(-half_disp_num, half_disp_num + 1).float()
    values = indices * (disp_range / half_disp_num)
    return values


if __name__ == "__main__":
    ind_saiA = torch.arange(5 * 5)[24]  # 0:an2-1
    import cv2
    import numpy

    ang_res = 5
    disp_range = 1
    disp_num = 10
    path = 'F:/hr_scene/'

    # x = generate_disparity_value(4,15)
    # print(15//2)

    img = cv2.imread(path + str(3).zfill(2) + '_' + str(3).zfill(2) + '.png').astype(np.float32) / 255.0
    # img = cv2.imread(path + 'MLIA.png').astype(np.float32) / 255.0
    # img = rearrange(img, '(h ah) (w aw) c -> (ah h) (aw w) c', ah=ang_res, aw=ang_res)
    img = torch.tensor(img).cuda().unsqueeze(dim=0)
    img = img[:,:,:,0:1].permute([0,3,1,2])
    #
    lbp = DifferentiableLBP(kernel_size=3, stride=1, alpha=10.0).cuda()
    x = lbp(img)
    print(x[0,0].shape)
    # print(x.max())
    # cv2.imwrite('lbp1.png', (x[0,0] * 255).cpu().numpy().astype(np.uint8))
    #
    # # y = center_to_boundary(x, ang_res, disp_range, disp_num)
    # net = PLFConstruct(1, ang_res, disp_range, disp_num)
    # z = net(img)
    # z = rearrange(z, '(b ah aw) c h w -> b c ah aw h w', ah=ang_res, aw=ang_res)
    # print(z.shape)
    #
    # out = z.squeeze().cpu().numpy()
    # for au in range(5):
    #     for av in range(5):
    #         cv2.imwrite(str(au+1).zfill(2)+'_'+str(av+1).zfill(2)+'.png', (out[0,au,av,:,:]*255).astype(np.uint8))

