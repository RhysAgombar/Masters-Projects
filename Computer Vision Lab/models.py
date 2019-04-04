import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import FlowNetSD


batch_size = 4

################################################################
## Define Networks 
################################################################

#layers=4, output_channel=3
## Generator Network
class genNet(nn.Module):
    def __init__(self):
        super(genNet, self).__init__() #3*3
        self.L1conv1 = nn.Conv2d(in_channels=3*batch_size, out_channels=64, kernel_size=3, padding=1)
        self.L1conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1) ## in list
        self.L1mp = nn.MaxPool2d(2)

        self.L2conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.L2conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1) ## in list
        self.L2mp = nn.MaxPool2d(2)

        self.L3conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.L3conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1) ## in list
        self.L3mp = nn.MaxPool2d(2)

        self.L4conv1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.L4conv2 = nn.Conv2d(512, 512, kernel_size=3, padding=1) ## in list

        self.L4deconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #nn.ConvTranspose2d(512, 256, 2)

        #self.L3concat = torch.cat((self.L4deconv, self.L3conv2)) #l4deconv, l3conv2
        self.L3Oconv1 = nn.Conv2d(256+512, 256, kernel_size=3, padding=1)
        self.L3Oconv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        self.L3deconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #nn.ConvTranspose2d(256, 128, 2)

        #self.L2concat = torch.cat((self.L3deconv, self.L2conv2)) #l3deconv, l2conv2
        self.L2Oconv1 = nn.Conv2d(128+256, 128, kernel_size=3, padding=1)
        self.L2Oconv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.L2deconv = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) #nn.ConvTranspose2d(128, 64, 2)

        #self.L1concat = torch.cat((self.L2deconv, self.L1conv2)) #l2deconv, l1conv2
        self.L1Oconv1 = nn.Conv2d(64+128, 64, kernel_size=3, padding=1)
        self.L1Oconv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.out_conv = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.out_act = nn.Tanh()
        
    def forward(self, input):
        x = self.L1conv1(input)
        Lx1 = self.L1conv2(x)
        x = self.L1mp(Lx1)

        x = self.L2conv1(x)
        Lx2 = self.L2conv2(x)
        x = self.L2mp(Lx2)

        x = self.L3conv1(x)
        Lx3 = self.L3conv2(x)
        x = self.L3mp(Lx3)

        x = self.L4conv1(x)
        x = self.L4conv2(x)
        x = self.L4deconv(x)

        x = torch.cat((x, Lx3), dim=1)

        x = self.L3Oconv1(x)
        x = self.L3Oconv2(x)
        x = self.L3deconv(x)

        x = torch.cat((x, Lx2), dim=1)

        x = self.L2Oconv1(x)
        x = self.L2Oconv2(x)
        x = self.L2deconv(x)

        x = torch.cat((x, Lx1), dim=1)

        x = self.L1Oconv1(x)
        x = self.L1Oconv2(x)
        x = self.out_conv(x)

        x = self.out_act(x)

        return x

## Discriminator Network (Re-implemented pix2pix tensorflow discriminator in pytorch)
class disNet(nn.Module):
    def __init__(self):
        super(disNet, self).__init__() #3*3
        self.num_layers = 4

        self.L1conv = nn.Conv2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=2)
        self.L2conv = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=2)
        self.L3conv = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=2)
        self.L4conv = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=4, stride=1, padding=2)
        
        self.logits = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=4, stride=1, padding=2)
        self.relu = nn.ReLU()
        self.predictions = nn.Sigmoid()
        
    def forward(self, input):
        x = self.relu(self.L1conv(input))
        x = self.relu(self.L2conv(x))
        x = self.relu(self.L3conv(x))
        x = self.relu(self.L4conv(x))
        logits = self.logits(x)
        predictions = self.predictions(logits)
        return logits, predictions

## Init Networks
def init_weights(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.xavier_uniform(m.weight)


## Flownet Model from https://github.com/NVIDIA/flownet2-pytorch
class FlowNet2SD(FlowNetSD.FlowNetSD):
    def __init__(self, args, batchNorm=False, div_flow=20):
        super(FlowNet2SD,self).__init__(args, batchNorm=batchNorm)
        self.rgb_max = args.rgb_max
        self.div_flow = div_flow

    def forward(self, inputs):
        rgb_mean = inputs.contiguous().view(inputs.size()[:2]+(-1,)).mean(dim=-1).view(inputs.size()[:2] + (1,1,1,))
        x = (inputs - rgb_mean) / self.rgb_max
        x = torch.cat((x[:,:,0,:,:], x[:,:,1,:,:]), dim = 1)

        out_conv0 = self.conv0(x)
        out_conv1 = self.conv1_1(self.conv1(out_conv0))
        out_conv2 = self.conv2_1(self.conv2(out_conv1))

        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6_1(self.conv6(out_conv5))

        flow6       = self.predict_flow6(out_conv6)
        flow6_up    = self.upsampled_flow6_to_5(flow6)
        out_deconv5 = self.deconv5(out_conv6)
        
        concat5 = torch.cat((out_conv5,out_deconv5,flow6_up),1)
        out_interconv5 = self.inter_conv5(concat5)
        flow5       = self.predict_flow5(out_interconv5)

        flow5_up    = self.upsampled_flow5_to_4(flow5)
        out_deconv4 = self.deconv4(concat5)
        
        concat4 = torch.cat((out_conv4,out_deconv4,flow5_up),1)
        out_interconv4 = self.inter_conv4(concat4)
        flow4       = self.predict_flow4(out_interconv4)
        flow4_up    = self.upsampled_flow4_to_3(flow4)
        out_deconv3 = self.deconv3(concat4)
        
        concat3 = torch.cat((out_conv3,out_deconv3,flow4_up),1)
        out_interconv3 = self.inter_conv3(concat3)
        flow3       = self.predict_flow3(out_interconv3)
        flow3_up    = self.upsampled_flow3_to_2(flow3)
        out_deconv2 = self.deconv2(concat3)

        concat2 = torch.cat((out_conv2,out_deconv2,flow3_up),1)
        out_interconv2 = self.inter_conv2(concat2)
        flow2 = self.predict_flow2(out_interconv2)

        #if self.training:
        #    return flow2,flow3,flow4,flow5,flow6
        #else:
        return self.upsample1(flow2*self.div_flow)
