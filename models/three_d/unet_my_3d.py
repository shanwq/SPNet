# 2D-Unet Model taken from https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import os


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''

    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True))

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InConv, self).__init__()
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(Up, self).__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch // 2, in_ch // 2, 2, stride=2)

        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class OutConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, input_channels, output_channels,
                 conv_op=nn.Conv3d, kernel_size=None, stride=None, padding = None,
                 norm_op=nn.InstanceNorm3d, norm_op_kwargs=None,
                 dropout_op=nn.Dropout3d, dropout_op_kwargs=None,
                 nonlin=nn.LeakyReLU, nonlin_kwargs=None):
        super(ConvBlock, self).__init__()
        if nonlin_kwargs is None:
            nonlin_kwargs = {'negative_slope': 1e-2, 'inplace': True}
        if dropout_op_kwargs is None:
            dropout_op_kwargs = {'p': 0.5, 'inplace': True}
        if norm_op_kwargs is None:
            norm_op_kwargs = {'eps': 1e-5, 'affine': True, 'momentum': 0.1}
        # if conv_kwargs is None:
        #     conv_kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1, 'bias': True}

        self.nonlin_kwargs = nonlin_kwargs
        self.nonlin = nonlin
        self.dropout_op = dropout_op
        self.dropout_op_kwargs = dropout_op_kwargs
        self.norm_op_kwargs = norm_op_kwargs
        # self.conv_kwargs = conv_kwargs
        self.conv_op = conv_op
        self.norm_op = norm_op
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = self.conv_op(input_channels, output_channels, kernel_size, stride, padding = padding)
        
        if self.dropout_op is not None and self.dropout_op_kwargs['p'] is not None and self.dropout_op_kwargs[
            'p'] > 0:
            self.dropout = self.dropout_op(**self.dropout_op_kwargs)
        else:
            self.dropout = None
        
        self.instnorm = self.norm_op(output_channels, **self.norm_op_kwargs)
        self.lrelu = self.nonlin(**self.nonlin_kwargs)

    def forward(self, x):
        x = self.conv(x)
        # print('conv.shape is', x.shape)
        if self.dropout is not None:
            x = self.dropout(x)
        return self.lrelu(self.instnorm(x))

class Unet_Mip3D(nn.Module):
    def __init__(self, in_channels, classes):
        super(Unet_Mip3D, self).__init__()
        self.n_channels = in_channels
        self.n_classes =  classes
        
        self.conv_blocks_context_0 = nn.Sequential(
            ConvBlock(input_channels=1, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
            ConvBlock(input_channels=32, output_channels=32, kernel_size=(1,3,3), stride=(1,1,1), padding = (0,1,1)),
        )
        self.conv_blocks_context_1= nn.Sequential(
            ConvBlock(input_channels=32, output_channels=64, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=64, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_2= nn.Sequential(
            ConvBlock(input_channels=64, output_channels=128, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=128, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_3= nn.Sequential(
            ConvBlock(input_channels=128, output_channels=256, kernel_size=(3,3,3), stride=(2,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_4= nn.Sequential(
            ConvBlock(input_channels=256, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.conv_blocks_context_5= nn.Sequential(
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,2,2), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_0 = nn.ConvTranspose3d(in_channels=320, out_channels=320, kernel_size=(1,2,2), 
                                       stride=(1,2,2), bias=False)
        self.conv_blocks_localization_0= nn.Sequential(
            ConvBlock(input_channels=640, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=320, output_channels=320, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_1 = nn.ConvTranspose3d(in_channels=320, out_channels=256, kernel_size=(1,2,2), 
                                       stride=(1,2,2), bias=False)
        self.conv_blocks_localization_1 = nn.Sequential(
            ConvBlock(input_channels=512, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=256, output_channels=256, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.tu_2 = nn.ConvTranspose3d(in_channels=256, out_channels=128, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)
        self.conv_blocks_localization_2 = nn.Sequential(
            ConvBlock(input_channels=256, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=128, output_channels=128, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_3 = nn.ConvTranspose3d(in_channels=128, out_channels=64, kernel_size=(2,2,2), 
                                       stride=(2,2,2), bias=False)

        self.conv_blocks_localization_3 = nn.Sequential(
            ConvBlock(input_channels=128, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=64, output_channels=64, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        
        self.tu_4 = nn.ConvTranspose3d(in_channels=64, out_channels=32, kernel_size=(1,2,2), 
                                       stride=(1,2,2), bias=False)

        self.conv_blocks_localization_4 = nn.Sequential(
            ConvBlock(input_channels=64, output_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
            ConvBlock(input_channels=32, output_channels=32, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1)),
        )
        self.pred_soft = ConvBlock(input_channels=32, output_channels=1, kernel_size=(3,3,3), stride=(1,1,1), padding = (1,1,1))
        self.softmax = nn.Softmax(dim=1)
        # ConvBlock(input_channels=32, output_channels=2, kernel_size=(3,3), stride=(1,1), padding = (1,1))
    def forward(self, x):
        features = []
        ds_feats = []
        conv0 = self.conv_blocks_context_0(x)
        # print('conv0.shape', conv0.shape)
        conv1 = self.conv_blocks_context_1(conv0)
        # print('conv1.shape', conv1.shape)
        conv2 = self.conv_blocks_context_2(conv1)
        # print('conv2.shape', conv2.shape)
        conv3 = self.conv_blocks_context_3(conv2)
        # print('conv3.shape', conv3.shape)
        conv4 = self.conv_blocks_context_4(conv3)
        # print('conv4.shape', conv4.shape)
        conv5 = self.conv_blocks_context_5(conv4)
        # print('conv5.shape', conv5.shape)
        # quit()
        deconv_5 = self.tu_0(conv5)
        deconv_5 = torch.cat((deconv_5, conv4), dim=1)
        deconv_5 = self.conv_blocks_localization_0(deconv_5)
        ds_feats.append(deconv_5)
        # print('conv_blocks_localization_0.shape is', deconv_5.shape)
        
        deconv_4 = self.tu_1(deconv_5)
        deconv_4 = torch.cat((deconv_4, conv3), dim=1)
        deconv_4 = self.conv_blocks_localization_1(deconv_4)
        ds_feats.append(deconv_4)
        # print('conv_blocks_localization_1.shape is', deconv_4.shape)
        
        deconv_3 = self.tu_2(deconv_4)
        deconv_3 = torch.cat((deconv_3, conv2), dim=1)
        deconv_3 = self.conv_blocks_localization_2(deconv_3)
        ds_feats.append(deconv_3)
        # print('conv_blocks_localization_2.shape is', deconv_3.shape)
        
        deconv_2 = self.tu_3(deconv_3)
        deconv_2 = torch.cat((deconv_2, conv1), dim=1)
        deconv_2 = self.conv_blocks_localization_3(deconv_2)
        ds_feats.append(deconv_2)
        # print('conv_blocks_localization_3.shape is', deconv_2.shape)
        
        deconv_1 = self.tu_4(deconv_2)
        deconv_1 = torch.cat((deconv_1, conv0), dim=1)
        deconv_1 = self.conv_blocks_localization_4(deconv_1)
        # print('conv_blocks_localization_4.shape is', deconv_1.shape)
        # print(x.shape)
        # quit()
        pred_soft = self.pred_soft(deconv_1)
        pred_soft = self.softmax(pred_soft)
    
        # feature = {'0':[conv0, deconv_1], '1':[conv1, deconv_2], '2':[conv2, deconv_3],
        #     '3':[conv3, deconv_4], '4':[conv4, deconv_5], '5':[conv5]}
        feature = [conv0, conv1, conv2, conv3, conv4, conv5, deconv_5, deconv_4, deconv_3, deconv_2]#, deconv_1]
        return pred_soft, feature
        # return x
        ''' conv0, conv1, conv2, conv3, conv4, conv5, deconv5, deconv4, deconv3, deconv2, deconv1 
        feature = {'0':[conv0, deconv_1], '1':[conv1, deconv_2], '2':[conv2, deconv_3],
            '3':[conv3, deconv_4], '4':[conv4, deconv_5], '5':[conv5]}
        '''

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES']='1'
    Unet_mip = Unet_Mip3D(1,2)
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    Unet_mip = Unet_mip.cuda()
    input_tensor = torch.randn(1, 1, 64, 128, 128).cuda()

    pred_soft, feat = Unet_mip(input_tensor)
    print(pred_soft.shape)
