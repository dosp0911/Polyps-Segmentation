
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchsummary import summary


class BnReLu(nn.Module):
    def __init__(self, in_num):
        super(BnReLu, self).__init__()

        self.sequential = nn.Sequential(
            nn.BatchNorm2d(in_num),
            nn.ReLU()
        )

    def forward(self, x):
        return self.sequential(x)


class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, stride=stride)

    def forward(self, x):
        return self.conv1(x)


class Block(nn.Module):
    def __init__(self, in_c, out_c, k_size, s1=1, s2=1, r_s=1, is_first=False, padding=1):
        super(Block, self).__init__()
        self.residual_block = ResidualBlock(in_c, out_c, stride=r_s)

        if is_first:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, stride=s1, padding=padding),
                BnReLu(out_c),
                nn.Conv2d(out_c, out_c, k_size, stride=s2, padding=padding),
            )
        else:
            self.sequential = nn.Sequential(
                BnReLu(in_c),
                nn.Conv2d(in_c, out_c, k_size, stride=s1, padding=padding),
                BnReLu(out_c),
                nn.Conv2d(out_c, out_c, k_size, stride=s2, padding=padding)
            )

    def forward(self, x):
        identity = self.residual_block(x)
        x = identity + self.sequential(x)
        return x


class GlobalAttentionUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GlobalAttentionUpsample, self).__init__()
        self.low_conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.low_bn_relu = BnReLu(out_channels)
        self.up_sample = nn.UpsamplingNearest2d(scale_factor=2)
        self.gapp = nn.AdaptiveAvgPool2d(1)
        self.high_conv = nn.Conv2d(out_channels, out_channels, 1)
        self.high_bn_relu = BnReLu(out_channels)

    def forward(self, low_x, high_x):
        identity_x = self.up_sample(high_x)
        low_x = self.low_conv(low_x)
        low_x = self.low_bn_relu(low_x)
        high_x = self.gapp(high_x)
        high_x = self.high_conv(high_x)
        high_x = self.high_bn_relu(high_x)

        high_x = low_x * high_x
        high_x += identity_x

        return high_x


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())

    def forward(self, x):
        size = x.shape[-2:]
        for mod in self:
            x = mod(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)


class ASPP(nn.Module):
    def __init__(self, in_c, out_c):
        super(ASPP, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                   nn.BatchNorm2d(out_c),
                                   nn.ReLU())

        self.aspp1 = ASPPConv(in_c, out_c, 6)
        self.aspp2 = ASPPConv(in_c, out_c, 12)
        self.aspp3 = ASPPConv(in_c, out_c, 18)
        self.avg_pool = ASPPPooling(in_c, out_c)

        self.conv_f = nn.Sequential(nn.Conv2d(5 * out_c, out_c, 1, bias=False),
                                    nn.BatchNorm2d(out_c),
                                    nn.ReLU(),
                                    nn.Dropout(0.5))

    def forward(self, x):
        out1 = self.conv1(x)
        out2 = self.aspp1(x)
        out3 = self.aspp2(x)
        out4 = self.aspp3(x)
        out5 = self.avg_pool(x)

        cat_t = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv_f(cat_t)
        return out


class SEBlock(nn.Module):
    def __init__(self, ch: int, ratio: int):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1) # shape: [N, C, 1, 1]
        self.extension = nn.Sequential(nn.Conv2d(ch, ch//ratio, 1), # shape: [N, C/r, 1, 1]
                                       nn.BatchNorm2d(ch//ratio),
                                       nn.ReLU(), # shape: [N, C/r, 1, 1]
                                       nn.Conv2d(ch//ratio, ch, 1), # shape: [N, C, 1, 1]
                                       nn.BatchNorm2d(ch),
                                       nn.Sigmoid()) # shape: [N, C, 1, 1]

    def forward(self, x):
        u = x
        x = self.squeeze(x)
        x = self.extension(x)
        return u * x


class ResUnetPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnetPP, self).__init__()

        self.enc_block1 = Block(in_channels, 64, 3, is_first=True)
        self.se_block1 = SEBlock(64, 16)
        self.enc_block2 = Block(64, 128, 3, 2, 1, 2)
        self.se_block2 = SEBlock(128, 16)
        self.enc_block3 = Block(128, 256, 3, 2, 1, 2)
        self.se_block3 = SEBlock(256, 16)
        self.enc_block4 = Block(256, 512, 3, 2, 1, 2)

        self.bridge = ASPP(512, 512)

        self.attention3 = GlobalAttentionUpsample(256, 512)
        # self.up_sample3 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec_block3 = Block(256 + 512, 256, 3)

        self.attention2 = GlobalAttentionUpsample(128, 256)
        # self.up_sample2 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec_block2 = Block(128 + 256, 128, 3)

        self.attention1 = GlobalAttentionUpsample(64, 128)
        # self.up_sample1 = nn.UpsamplingNearest2d(scale_factor=2)
        self.dec_block1 = Block(64 + 128, 64, 3)

        self.final_layer = nn.Sequential(
                                ASPP(64, 64),
                                nn.Conv2d(64, out_channels, 1),
                                nn.Sigmoid())

        self.init_weights()

    def forward(self, x):
        # x : shape:[N, 3, H, W]
        ## encoder
        enc_out1 = self.enc_block1(x) # shape:[N, 64, H, W]
        enc_out1_se = self.se_block1(enc_out1) # shape:[N, 64, H, W]
        enc_out2 = self.enc_block2(enc_out1_se) # shape:[N, 128, H/2, W/2]
        enc_out2_se = self.se_block2(enc_out2) # shape:[N, 128, H/2, W/2]
        enc_out3 = self.enc_block3(enc_out2_se) # shape:[N, 256, H/4, W/4]
        enc_out3_se = self.se_block3(enc_out3) # shape:[N, 256, H/4, W/4]
        enc_out4 = self.enc_block4(enc_out3_se) # shape:[N, 512, H/8, W/8]

        ## bridge
        x = self.bridge(enc_out4) # shape:[N, 512, H/8, W/8]

        ## decoder
        x = self.attention3(enc_out3, x) # shape:[N, 512, H/4, W/4]
        # x = self.up_sample3(x)
        x = torch.cat([enc_out3, x], dim=1) # shape:[N, 256+512, H/4, W/4]
        x = self.dec_block3(x) # shape:[N, 256, H/4, W/4]

        x = self.attention2(enc_out2, x) # shape:[N, 256, H/2, W/2]
        # x = self.up_sample2(x)
        x = torch.cat([enc_out2, x], dim=1)
        x = self.dec_block2(x)

        x = self.attention1(enc_out1, x)
        # x = self.up_sample1(x)
        x = torch.cat([enc_out1, x], dim=1)
        x = self.dec_block1(x)

        x = self.final_layer(x)

        return x

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=1e-3)
                if m.bias != None:
                    nn.init.constant_(m.bias, 0)


if __name__ =="__main__":
    from torchsummary import summary
    from torch.utils.tensorboard.writer import SummaryWriter
    a = torch.rand((3, 3, 128, 128))
    model = ResUnetPP(3, 1)
    # summary(model, (3,128,128), device='cpu')
    writer = SummaryWriter('./')
    writer.add_graph(model, a)
    writer.close()