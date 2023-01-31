############# PS-LH1  #######################################
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        self.other = other
        """
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   64, k=3, stride=1, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 64,  128, k=3, stride=2, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

        """
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,    64, k=3, stride=1, pad=1)   
        self.conv3 = model_utils.conv(batchNorm, 64,    64, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)#128*16*16# #################
        self.conv5 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)                            #
        self.conv6 = model_utils.conv(batchNorm, 128,  256, k=3, stride=1, pad=1)#256*16*16##############    #
        self.conv7 = model_utils.conv(batchNorm, 256,  256, k=3, stride=2, pad=1)#256*8*8############   #    #
        self.conv8 = model_utils.conv(batchNorm, 256,  512, k=3, stride=1, pad=1)#512*8*8#########  #   #    #
        self.conv9 = model_utils.conv(batchNorm, 512,  512, k=3, stride=1, pad=1)#512*8*8######  #  #   #    #
        self.conv10= model_utils.conv(batchNorm, 512, 1028, k=3, stride=1, pad=1)#1028*8*8    #  #  #   #    #
        self.conv11= model_utils.conv(batchNorm, 1028,1028, k=3, stride=1, pad=1)             #  #  #   #    #
        self.conv12= model_utils.conv(batchNorm, 1028, 512, k=3, stride=1, pad=1)#512*8*8######  #  #   #    #
        self.conv121= model_utils.conv(batchNorm, 512, 512, k=3, stride=1, pad=1)                #  #   #    #
        self.conv13= model_utils.conv(batchNorm, 512,  256, k=3, stride=1, pad=1)#256*8*8############   #    #
        self.conv14= model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)                #      #    #
                                                                                                 #      #    #
        self.conv15= model_utils.conv(batchNorm, 256,  512, k=3, stride=1, pad=1)#512*8*8#########      #    #
        self.conv16= model_utils.conv(batchNorm, 512,  512, k=3, stride=1, pad=1)                       #    #
        self.conv17= model_utils.deconv(512, 256)                                #256*16*16##############    #
        self.conv18= model_utils.conv(batchNorm, 256,  128, k=3, stride=1, pad=1)#128*16*16###################
        self.conv19= model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)

        self.lrelu = model_utils.LReLU1()



    def forward(self, x):
        out = self.conv1(x)#064*32*32
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)#128*16*16 ##################################
        identity1 = out                                                  #
        out = self.conv5(out)                                            #
        out = self.conv6(out)#256*16*16 ##############################   #
        identity2 = out                                              #   #
        out = self.conv7(out)#256*8*8############################    #   #
        identity3 = out                                         #    #   #
        out = self.conv8(out)#512*8*8 ########################  #    #   #
        identity4 = out                                      #  #    #   #
        out = self.conv9(out)#512*8*8 ###############        #  #    #   #
        identity5 = out                             #        #  #    #   #
        out = self.conv10(out)#1028*8*8             #        #  #    #   #
        out = self.conv11(out)                      #        #  #    #   #
        out = self.conv12(out)#512*8*8###############        #  #    #   #
        out = out + identity5                                #  #    #   #
        out = self.lrelu(out)                                #  #    #   #
        out = self.conv13(out)#256*8*8 ##########################    #   #
        out = out + identity3                                #       #   #
        out = self.lrelu(out)                                #       #   #
        out = self.conv14(out)                               #       #   #
        out = self.conv15(out)#512*8*8 #######################       #   #
        out = out + identity4                                        #   #
        out = self.lrelu(out)                                        #   #
        out = self.conv16(out)                                       #   #
        out = self.conv17(out)#256*16*16  ############################   #
        out = out + identity2                                            #
        out = self.lrelu(out)                                            #
        out = self.conv18(out)#128*16*16 #################################
        out = out + identity1                                            
        out = self.lrelu(out)                                            
        out_feat = self.conv19(out)
        
        n, c, h, w = out_feat.data.shape

        out_feat   = out_feat.view(-1)
        return out_feat, [n, c, h, w]


class Regressor(nn.Module):
    def __init__(self, batchNorm=False, other={}): 
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_normal= self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
               nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        normal = self.est_normal(out)
        normal = torch.nn.functional.normalize(normal, 2, 1)
        
        return normal

class PS_FCN(nn.Module):
    def __init__(self, fuse_type='max', batchNorm=False, c_in=3, other={}):
        super(PS_FCN, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
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
            feat, shape = self.extractor(net_in)
            #print('#################shapeee###############################################')
            #print(shape.shape)
            #print('#########################################################################################')
            feats.append(feat)
        #print('#################shapeee###############################################')
        #print(shape)
        #print('#########################################################################################')
        #print('###############feat   ###########################################')
        #print(feat.shape)
        #print('#########################################################################################')
        #print('###############len   ###########################################')
        #print(len(img_split))
        #print('#########################################################################################')
        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        #print('###############self fused  ###########################################')
        #print(feat_fused.shape )
        #print('#########################################################################################')
        normal = self.regressor(feat_fused, shape)
        #print('################normal = out var ######################################')
        #print( normal.shape)
        #print('#########################################################################################')
        return normal

