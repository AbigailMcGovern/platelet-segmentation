import numpy as np
import torch 
import torch.nn as nn
import torch.nn.functional as F

# convolution module
class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding0=1, padding1=1, final='relu'):
        super(ConvModule, self).__init__()

        # Convolutions
        # ------------
        self.conv0 = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding0)
        self.conv1 = nn.Conv3d(out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding1)

        # Batch Normailsation
        # -------------------
        self.batch0 = nn.BatchNorm3d(out_channels)
        self.batch1 = nn.BatchNorm3d(out_channels)

        # Activation
        # ----------
        self.relu0 = nn.ReLU()
        self.relu1 = nn.ReLU()
        self.sm = nn.Softmax()
        self.final = final


    def forward(self, x):
        # First convolution
        x = self.conv0(x)
        x = self.batch0(x)
        x = self.relu0(x)
        x = self.conv1(x)
        x = self.batch1(x)
        if self.final == 'relu':
            x = self.relu1(x)
        elif self.final == 'softmax':
            x = self.sm(x)
        return x


# trial UNet
class UNet(nn.Module):

    def __init__(self, in_channels=1, out_channels=3, down_factors=(1, 2, 2), up='convolution'):
        '''
        in_channels: int
        out_channels: int
        down_factors: tuple of int
            Factors by which to downsample in encoder
        up: str
            'bilinear': use bilinear/nearest neighbour interpolations for up sampling in decoder
            'convolution': use inverse convolutions with learnable parameters
        '''
        super(UNet, self).__init__()

        # Max pooling 
        # -----------
        # encoder: downsample 4 times
        # Get the padding for the max pool
        #   Must be at most half of 
        p = [np.floor_divide(df, 2).astype(int) for df in down_factors]
        p = tuple(p)
        # max pool layers
        self.d0 = nn.MaxPool3d(
                               down_factors, 
                               stride=down_factors, 
                               padding=(0, 1, 1)
                               )
        self.d1 = nn.MaxPool3d(
                               down_factors, 
                               stride=down_factors, 
                               padding=p
                               )
        self.d2 = nn.MaxPool3d(
                               down_factors, 
                               stride=down_factors, 
                               padding=(0, 1, 1)
                               )
        self.d3 = nn.MaxPool3d(
                               down_factors, 
                               stride=down_factors, 
                               padding=(0, 1, 1)
                               )

        # Convolutions
        # ------------
        # encoder colvolutions:
        self.c0 = ConvModule(in_channels, 32)
        self.c1 = ConvModule(32, 64)
        self.c2 = ConvModule(64, 128)
        self.c3 = ConvModule(128, 256)
        self.c4 = ConvModule(256, 256)

        # decoder convolutions
        self.c5 = ConvModule(256 * 2, 128)
        self.c6 = ConvModule(128 * 2, 64)
        self.c7 = ConvModule(64 * 2, 32)
        self.c8 = ConvModule(32 * 2, out_channels, final='softmax')

        # Upsampling
        # ----------
        # Inverse convolutions
        if up == 'convolution':
            self.up0 = nn.ConvTranspose3d(
                                          256, 
                                          256, 
                                          kernel_size=down_factors, 
                                          stride=down_factors, 
                                          groups=256)
            self.up1 = nn.ConvTranspose3d(
                                          128, 
                                          128, 
                                          kernel_size=down_factors, 
                                          stride=down_factors, 
                                          groups=128
                                          )
            self.up2 = nn.ConvTranspose3d(
                                          64, 
                                          64, 
                                          kernel_size=down_factors, 
                                          stride=down_factors, 
                                          groups=64
                                          )
            self.up3 = nn.ConvTranspose3d(
                                          32, 
                                          32, 
                                          kernel_size=down_factors, 
                                          stride=down_factors, 
                                          groups=32
                                          )
        elif up == 'bilinear':
            self.up0 = lambda x: F.interpolate(
                                               x, 
                                               mode='tconv', 
                                               scale_factor=down_factors
                                               )
            self.up1 = lambda x: F.interpolate(
                                               x, 
                                               mode='tconv', 
                                               scale_factor=down_factors
                                               )
            self.up2 = lambda x: F.interpolate(
                                               x, 
                                               mode='tconv', 
                                               scale_factor=down_factors
                                               )
            self.up3 = lambda x: F.interpolate(
                                               x, 
                                               mode='tconv', 
                                               scale_factor=down_factors
                                               )

        # padding=(0, 1, 1)

        # Final Softmax
        # -------------
        #self.sm = nn.Softmax()

    def forward(self, x):
        # Encoder
        # -------
        c0 = self.c0(x)
        x = self.d0(c0)
        c1 = self.c1(x)
        x = self.d1(c1)
        c2 = self.c2(x)
        x = self.d2(c2)
        c3 = self.c3(x)
        x = self.d3(c3)
        x = self.c4(x)

        # Decoder
        # -------
        x = self.up0(x)
        # quick dumb hack for concatenation 
        x = x[:, :, :, :-1, :-1]
        x = torch.cat([x, c3], 1)
        x = self.c5(x)
        x = self.up1(x)
        x = x[:, :, :, :-1, :-1]
        x = torch.cat([x, c2], 1)
        x = self.c6(x)
        x = self.up2(x)
        x = x[:, :, :, :-1, :-1]
        x = torch.cat([x, c1], 1)
        x = self.c7(x)
        x = self.up3(x)
        x = x[:, :, :, 1:-1, 1:-1]
        x = torch.cat([x, c0], 1)
        x = self.c8(x)
        #x = self.sm(x)
        return x

if __name__ == '__main__':
    ip = torch.randn(1, 1, 10, 256, 256)
    unet = UNet()
    o = unet(ip)
    # output has shape (1, 3, 10, 256, 256)

    # c0 -> torch.Size([1, 32, 10, 256, 256])
    # d0, c1 -> torch.Size([1, 64, 10, 129, 129])
    # d1, c2 -> torch.Size([1, 128, 10, 65, 65])
    # d2, c3 -> torch.Size([1, 256, 10, 33, 33])
    # d3, c4 -> torch.Size([1, 256, 10, 17, 17])
    # u0 -> torch.Size([1, 256, 10, 33, 33])
    # u1, c5 -> torch.Size([1, 128, 10, 65, 65])
    # u2, c6 -> torch.Size([1, 64, 10, 129, 129])
    # u3, c7 -> torch.Size([1, 32, 10, 256, 256])
    # c8 -> torch.Size([1, 3, 10, 256, 256])