
import torch
import torch.nn as nn
from torchsummary import summary


class Con2D(nn.Module):
    def __init__(self, in_c, out_c, k_size, padding=1, is_bn=True):
        super(Con2D, self).__init__()

        if is_bn:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, padding=padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size, padding=padding),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )
        else:
            self.sequential = nn.Sequential(
                nn.Conv2d(in_c, out_c, k_size, padding=padding),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, k_size, padding=padding),
                nn.ReLU(inplace=True)
            )

    def forward(self, x):
        return self.sequential(x)


class Unet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unet, self).__init__()

        self.con_block_1 = Con2D(in_channels, 64, 3)
        self.con_block_2 = Con2D(64, 128, 3)
        self.con_block_3 = Con2D(128, 256, 3)
        self.con_block_4 = Con2D(256, 512, 3)
        self.con_block_5 = Con2D(512, 1024, 3)

        self.exp_block_4 = Con2D(1024, 512, 3, is_bn=False)
        self.exp_block_3 = Con2D(512, 256, 3, is_bn=False)
        self.exp_block_2 = Con2D(256, 128, 3, is_bn=False)
        self.exp_block_1 = Con2D(128, 64, 3, is_bn=False)

        self.deconv_4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.deconv_3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.deconv_2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.deconv_1 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.final_layer = nn.Conv2d(64, out_channels, 1)

        self.init_weights()

    def forward(self, x):
        con_block_1_out = self.con_block_1(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_1_out)
        con_block_2_out = self.con_block_2(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_2_out)
        con_block_3_out = self.con_block_3(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_3_out)
        con_block_4_out = self.con_block_4(x)
        x = nn.MaxPool2d(2, stride=2)(con_block_4_out)
        x = self.con_block_5(x)

        x = self.deconv_4(x)
        x = torch.cat([con_block_4_out, x], dim=1)
        x = self.exp_block_4(x)

        x = self.deconv_3(x)
        x = torch.cat([con_block_3_out, x], dim=1)
        x = self.exp_block_3(x)

        x = self.deconv_2(x)
        x = torch.cat([con_block_2_out, x], dim=1)
        x = self.exp_block_2(x)

        x = self.deconv_1(x)
        x = torch.cat([con_block_1_out, x], dim=1)
        x = self.exp_block_1(x)

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
    u_net = Unet(1, 4)
    summary(u_net, (1, 256, 1600), device='cpu')
