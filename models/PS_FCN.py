#############amelioration PS-FA#######################################
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
        #elf.conv2 = model_utils.conv(batchNorm, 64,    64, k=3, stride=1, pad=1)   
        #elf.conv3 = model_utils.conv(batchNorm, 64,    64, k=3, stride=1, pad=1)
        self.conv4 = model_utils.convS(batchNorm,  64,  128, k=3, stride=2, pad=1)#128*16*16# #################
        self.conv5 = model_utils.convS(batchNorm, 128,  128, k=3, stride=1, pad=1)#128*16*16##############    #
        self.conv6 = model_utils.convS(batchNorm, 128,  256, k=3, stride=2, pad=1)#256*8*8############   #    #
        #elf.conv7 = model_utils.convS(batchNorm, 256,  256, k=3, stride=2, pad=1)                   #   #    #
        self.conv8 = model_utils.conv(batchNorm,  256,  256, k=3, stride=1, pad=1)#256*8*8#########  #   #    #
        #elf.conv9 = model_utils.convS(batchNorm, 512,  512, k=3, stride=1, pad=1)#512*8*8######  #  #   #    #
        #self.conv10= model_utils.conv(batchNorm, 512, 1028, k=3, stride=1, pad=1)#1028*8*8    #  #  #   #    #
        #elf.conv11= model_utils.conv(batchNorm, 1028,1028, k=3, stride=1, pad=1)              #  #  #   #    #
        #elf.conv12= model_utils.conv(batchNorm, 1028, 512, k=3, stride=1, pad=1)#512*8*8#######  #  #   #    #
        self.conv13= model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)#256*8*8#############   #    #
        #elf.conv14= model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)                 #      #    #
                                                                                                  #      #    #
        #elf.conv15= model_utils.conv(batchNorm, 256,  512, k=3, stride=1, pad=1)#512*8*8##########      #    #
        #elf.conv16= model_utils.conv(batchNorm, 512,  512, k=3, stride=1, pad=1)                        #    #
        self.conv17= model_utils.deconv(256, 128)                                #128*16*16###############    #
        self.conv18= model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)#128*16*16####################
        self.conv19= model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)

        self.lrelu = model_utils.LReLU1()



    def forward(self, x):
        out = self.conv1(x)#064*32*32
        #out = self.conv2(out)
        #out = self.conv3(out)
        out = self.conv4(out)#128*16*16 ##################################
        identity1 = out                                                  #
        out = self.lrelu(out)                                            #
        out = self.conv5(out)#128*16*16 ##############################   #
        identity2 = out                                              #   #
        out = self.lrelu(out)                                        #   #
        out = self.conv6(out)#256*8*8############################    #   #
        identity3 = out                                         #    #   #
        out = self.lrelu(out)                                   #    #   #
        #ut = self.conv7(out)                                   #    #   #
        out = self.conv8(out)#512*8*8 ########################  #    #   #
        #ut = self.conv9(out)#512*8*8 ###############        #  #    #   #
        #dentity5 = out                             #        #  #    #   #
        #ut = self.lrelu(out)                       #        #  #    #   #
        #ut = self.conv10(out)#1028*8*8             #        #  #    #   #
        #ut = self.conv11(out)                      #        #  #    #   #
        #ut = self.conv12(out)#512*8*8###############        #  #    #   #
        #ut = out + identity5                                #  #    #   #
        #ut = self.lrelu(out)                                #  #    #   #
        out = self.conv13(out)#256*8*8 ##########################    #   #
        out = out + identity3                                #       #   #
        out = self.lrelu(out)                                #       #   #
        #ut = self.conv14(out)                               #       #   #
        #ut = self.conv15(out)#512*8*8 #######################       #   #
        #ut = out + identity4                                        #   #
        #ut = self.lrelu(out)                                        #   #
        #ut = self.conv16(out)                                       #   #
        out = self.conv17(out)#128*16*16  ############################   #
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
        self.deconv4 = model_utils.conv(batchNorm, 64, 64,  k=3, stride=1, pad=1)
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
        out    = self.deconv4(out)
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
            feats.append(feat)
        if self.fuse_type == 'mean':
            feat_fused = torch.stack(feats, 1).mean(1)
        elif self.fuse_type == 'max':
            feat_fused, _ = torch.stack(feats, 1).max(1)
        normal = self.regressor(feat_fused, shape)
        return normal