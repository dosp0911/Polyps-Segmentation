
import torch
import torch.nn as nn
from torchsummary import summary


class BnReLu(nn.Module):
    def __init__(self, in_num):
        super(BnReLu, self).__init__()

        self.sequential = nn.Sequential(
            nn.BatchNorm2d(in_num),
            nn.ReLU(inplace=True)
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


class ResUnet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResUnet, self).__init__()

        self.enc_block1 = Block(in_channels, 64, 3, is_first=True)
        self.enc_block2 = Block(64, 128, 3, 2, 1, 2)
        self.enc_block3 = Block(128, 256, 3, 2, 1, 2)

        self.bridge = Block(256, 512, 3, 2, 1, 2)

        self.up_sample3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up_sample1 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.dec_block3 = Block(512+256, 256, 3)
        self.dec_block2 = Block(256+128, 128, 3)
        self.dec_block1 = Block(128+64, 64, 3)

        self.final_layer = nn.Conv2d(64, out_channels, 1)

        self.init_weights()

    def forward(self, x):

        ## encoder
        enc_out1 = self.enc_block1(x)
        enc_out2 = self.enc_block2(enc_out1)
        enc_out3 = self.enc_block3(enc_out2)

        ## bridge
        x = self.bridge(enc_out3)

        ## decoder
        x = self.up_sample3(x)
        x = torch.cat([enc_out3, x], dim=1)
        x = self.dec_block3(x)

        x = self.up_sample2(x)
        x = torch.cat([enc_out2, x], dim=1)
        x = self.dec_block2(x)

        x = self.up_sample1(x)
        x = torch.cat([enc_out1, x], dim=1)
        x = self.dec_block1(x)

        x = self.final_layer(x)
        x = nn.Sigmoid()(x)

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
    # writer = SummaryWriter('model')
    resunet = ResUnet(3, 1)
    print(resunet.__class__.__name__)
    # x = torch.rand((1, 3, 224, 224), dtype=torch.float, requires_grad=False)
    # writer.add_graph(resunet, x)
    # # out = resunet(x)
    # writer.close()
    # summary(resunet, (3, 256, 256), device='cpu')