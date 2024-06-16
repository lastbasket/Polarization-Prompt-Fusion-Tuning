# -- pytorch stuff --
import torch
import torch.nn as nn


def conv_bn_relu(ch_in, ch_out, kernel, stride=1, padding=0, bn=True,
                 relu=True):
    assert (kernel % 2) == 1, \
        'only odd kernel is supported but kernel = {}'.format(kernel)

    layers = []
    layers.append(nn.Conv2d(ch_in, ch_out, kernel, stride, padding,
                            bias=not bn))
    if bn:
        layers.append(nn.BatchNorm2d(ch_out))
    if relu:
        layers.append(nn.ReLU(inplace=True))

    layers = nn.Sequential(*layers)

    return layers


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, qk):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = qk.shape
        qk = qk.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(qk * self.smooth)
        else:
            mask = self.softmax(qk)
        output = mask * qk
        output = output.contiguous().view(b, c, h, w)

        return output


class ModalityPromper(nn.Module):
    def __init__(self, in_dim):
        super(ModalityPromper, self).__init__()
        self.qkv = nn.Linear(in_dim*2, in_dim*3, bias=False)
        qkv_total = sum([param.nelement() for param in self.qkv.parameters()])
        # print('qkv parameter: % .4fM' % (qkv_total / 1e6))
        self.relu = nn.ReLU(inplace=True)
        self.spa_conv = conv_bn_relu(in_dim*2, in_dim, kernel=1, stride=1)
        spa_conv_total = sum([param.nelement() for param in self.spa_conv.parameters()])
        # print('spa_conv parameter: % .4fM' % (spa_conv_total / 1e6))
        self.spa_softmax = nn.Softmax(dim=-1)
        self.smooth = nn.Parameter(torch.zeros(1) + 10.0)
        self.spa_out_drop = nn.Dropout(0.15, inplace=True)
        self.spa_proj = nn.Linear(in_dim, in_dim*2, bias=False)
        
        spa_proj_total = sum([param.nelement() for param in self.spa_proj.parameters()])
        # print('spa_proj parameter: % .4fM' % (spa_proj_total / 1e6))
        self.global_pool=nn.AdaptiveAvgPool2d(1)
        self.chan_softmax = nn.Softmax(dim=-1)
        self.chan_fc=nn.Conv2d(in_dim,in_dim*2,1, bias=False)
        
        chan_fc_total = sum([param.nelement() for param in self.chan_fc.parameters()])
        # print('chan_fc parameter: % .4fM' % (chan_fc_total / 1e6))
        self.chan_proj = nn.Linear(in_dim, in_dim, bias=False)
        
        chan_proj_total = sum([param.nelement() for param in self.chan_proj.parameters()])
        # print('spa_proj parameter: % .4fM' % (chan_proj_total / 1e6))


    def forward(self, x, prompt):
        B, C, H, W = x.shape
        qkv = self.relu(self.qkv(torch.concat([prompt, x], dim=1).reshape(B, 2*C, H*W).permute(0, 2, 1)))
        q = qkv[..., :C].permute(0,2,1).reshape(B, C, H, W)
        k = qkv[..., C:2*C].permute(0,2,1).reshape(B, C, H, W)
        v = qkv[..., 2*C:].permute(0,2,1).reshape(B, C, H, W)
        spa_embed = self.spa_conv(torch.concat([q,k], dim=1)).reshape(B, C, H*W)
        spa_att = (self.spa_softmax(spa_embed) * self.smooth).permute(0, 2, 1)
        
        # B, H*W, C -> B, H*W, 2*C -> B, C, H, W
        spa_feat = self.spa_out_drop(self.spa_proj(spa_att*v.reshape(B, C, H*W).permute(0, 2, 1)))
        spa_prompt = spa_feat[..., :C].permute(0,2,1).reshape(B, C, H, W)
        spa_out = x+spa_feat[..., C:].permute(0,2,1).reshape(B, C, H, W)

        # B, C, H, W -> B, H*W, C
        chan_q = prompt.reshape(B, C, H*W).permute(0, 2, 1)
        chan_k = x.reshape(B, C, H*W).permute(0, 2, 1)
        chan_v = k.reshape(B, C, H*W).permute(0, 2, 1)

        # B, C, H, W -> B, 2*C, 1, 1
        chan_att = self.chan_fc(self.global_pool(prompt+x))

        # B, C, 1, 1 -> B, 1, C
        chan_att_q = chan_att[:, :C].reshape(B, C, 1*1).permute(0, 2, 1)
        chan_att_k = chan_att[:, C:].reshape(B, C, 1*1).permute(0, 2, 1)

        # B, H*W, C -> B, C, H, W
        chan_prompt = self.chan_proj(chan_v*self.chan_softmax(chan_k*chan_att_k+chan_q*chan_att_q)).permute(0,2,1).reshape(B, C, H, W)
        # chan_prompt = (chan_v*self.chan_softmax(chan_k*chan_att_k+chan_q*chan_att_q)).permute(0,2,1).reshape(B, C, H, W)

        update_prompt = spa_prompt+chan_prompt

        return spa_out, update_prompt

