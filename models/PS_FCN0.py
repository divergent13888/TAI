#MODEL N01 resnet 
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

class block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1):
        super(block, self).__init__()

        self.expansion = 1
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(intermediate_channels)
        self.conv2 = nn.Conv2d( intermediate_channels, intermediate_channels, kernel_size=1, stride=stride, padding=0)
        self.bn2 = nn.BatchNorm2d(intermediate_channels)
        self.conv3 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(intermediate_channels * self.expansion)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()
        #print("################# x shape init block  ##############################")
        #print(x.shape)
        #print("######################################################## ")
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("################# x shape block conv1  ##############################")
        #print(x.shape)
        #print("######################################################## ")
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        #print("################# x shape block conv2  ##############################")
        #print(x.shape)
        #print(x.shape)
        #print("######################################################## ")
        x = self.conv3(x)
        x = self.bn3(x)
        #print("################# x shape block conv3 non lrelu  ##############################")
        #print(x.shape)
        #print("######################################################## ")
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x


class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels = 6):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(image_channels, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        #self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Essentially the entire ResNet architecture are in these 4 lines below
        self.layer1 = self._make_layer(
            block, layers[0], intermediate_channels=64, stride=1
        )
        self.layer2 = self._make_layer(
            block, layers[1], intermediate_channels=128, stride=2
        )
        self.layer3 = self._make_layer(
            block, layers[2], intermediate_channels=256, stride=1
        )
        self.layer4 = self._make_layer(
            block, layers[3], intermediate_channels=512, stride=1
        )
        self.conv2 = model_utils.conv(batchNorm =False, cin=512, cout=256,  k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm=False, cin= 256, cout= 128,  k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm=False, cin=128, cout=128,  k=3, stride=1, pad=1)
        self.avgpool = nn.AdaptiveAvgPool2d((16,16))
        #self.fc = nn.Linear(512 , num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("################# x shape 1st convulotion ##############################")
        #print(x.shape)
        #print(x.shape)
        #print("######################################################## ")
        #x = self.maxpool(x)
        #print("################# x shape 1st layer after maxpool ##############################")
        #print(x.shape)
        #print("######################################################## ")
        #print("################# start layer1 ##############################")
        x = self.layer1(x)
        #print("################# x shape layer1 ##############################")
        #print(x.shape)
        #print("######################################################## ")
        #print("################# start layer2 ##############################")
        x = self.layer2(x)
        #print("################# x shape layer2 ##############################")
        #print(x.shape)
        #print("######################################################## ")
        x = self.layer3(x)
        #print("################# x shape layer3 ##############################")
        #print(x.shape)
        #print("######################################################## ")
        x = self.layer4(x)
        #print("################# x shape layer4 ##############################")
        #print(x.shape)
        #print("######################################################## ")
      
        #print("################# x shape conv4 ##############################")
        #print(x.shape)
        #print("######################################################## ")
        #print("################# x shape avrgpool ##############################")
        #print(x.shape)
        #print("######################################################## ")
        #x = x.reshape(x.shape[0], -1)
        #print("################# x shape reshape ##############################")
        #print(x.shape)
        #print("######################################################## ")
        #x = self.fc(x)
        #print("################# x shape finale ##############################")
        #print(x.shape)
        #print("######################################################## ")
        out_feat = self.avgpool(x)
        n, c, h, w = out_feat.data.shape
        
        out_feat   = out_feat.view(-1)
        
        return out_feat, [n, c, h, w]
        

    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride):
        identity_downsample = None
        layers = []
        # Either if we half the input space for ex, 56x56 -> 28x28 (stride=2), or channels changes
        # we need to adapt the Identity (skip connection) so it will be able to be added
        # to the layer that's ahead
        if stride != 1 or self.in_channels != intermediate_channels :
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, intermediate_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(intermediate_channels),
            )

        layers.append(
            block(self.in_channels, intermediate_channels, identity_downsample, stride)
        )

        # The expansion size is always 4 for ResNet 50,101,152
        self.in_channels = intermediate_channels 

        # For example for first resnet layer: 256 will be mapped to 64 as intermediate layer,
        # then finally back to 256. Hence no identity downsample is needed, since stride = 1,
        # and also same amount of channels.
        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels))

        return nn.Sequential(*layers)


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}): 
        super(Regressor, self).__init__()
        self.other   = other
        self.conv1 = model_utils.conv(batchNorm, 512, 256,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 256, 256,  k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 256, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.conv1(x)
        out    = self.conv2(out)
        out    = self.conv3(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        return normal

class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.net = ResNet(block, [2, 2, 2, 2])
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        img   = x[0]
        img_split = torch.split(img, 3, 1)
        if len(x) > 1: # Have lighting
            light = x[1]
            light_split = torch.split(light, 3, 1)

        feats = []
        for i in range(len(img_split)):
            net_in = img_split[i] if len(x) == 1 else torch.cat([img_split[i], light_split[i]], 1)
            feat, shape = self.net(net_in)
            feats.append(feat)
        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        normal = self.regressor(feat_fused, shape)
        return normal
