# -- pytorch stuff --
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
# -- misc. utilities --
from functools import partial
# -- model imports --
from model.completionformer.resnet_cbam import BasicBlock
from .modality_promper import ModalityPromper
# -- mmcv stuff --
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint

model_path = {
    'resnet18': 'ckpts/resnet18.pth',
    'resnet34': 'ckpts/resnet34.pth'
}


def get_resnet18(pretrained=True):
    net = torchvision.models.resnet18(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet18'])
        net.load_state_dict(state_dict)

    return net


def get_resnet34(pretrained=True):
    net = torchvision.models.resnet34(pretrained=False)
    if pretrained:
        state_dict = torch.load(model_path['resnet34'])
        net.load_state_dict(state_dict)

    return net


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.resblock = BasicBlock(dim, dim, ratio=16)
        self.concat_conv = nn.Conv2d(dim*2, dim, kernel_size=(3, 3), padding=(1, 1), bias=False)


    def forward(self, x, H, W):
        input = x

        # Transformer branch
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        # CNN branch
        B, N, C = input.shape
        _, _, Cx = x.shape
        input = input.transpose(1, 2).view(B, C, H, W)
        input = self.resblock(input)

        # fusion
        x = x.transpose(1, 2).view(B, Cx, H, W)
        x = self.concat_conv(torch.cat([x, input], dim=1))
        x = x.flatten(2).transpose(1, 2)


        return x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, pretrained=None, use_prompt=False, foundation=None):
        super().__init__()
        self.depths = depths
        self.num_stages = num_stages
        self.use_prompt = use_prompt

        setattr(self, "embed_layer1", foundation.embed_layer1)
        setattr(self, "embed_layer2", foundation.embed_layer2)
        self.embed_layer1.requires_grad = False
        self.embed_layer2.requires_grad = False

        # in_chans = 128

        cur = 0

        chs = [128, 64, 128, 320, 512]

        heights = [104, 52, 26, 13, 6]
        widths = [136, 68, 34, 17, 8]

        if use_prompt:
            self.prompt_modifier0 = nn.Sequential(nn.Conv2d(48, chs[0], kernel_size=3, stride=2, padding=1, bias=False), \
                                                nn.ReLU(inplace=True), nn.BatchNorm2d(chs[0]))


            HID_DIM = 256

            self.mp0 = ModalityPromper(chs[0])
            total = sum([param.nelement() for param in self.mp0.parameters()]) + sum([param.nelement() for param in self.prompt_modifier0.parameters()])
            # print('P0 parameter: % .4fM' % (total / 1e6))

        for i in range(num_stages):
            if use_prompt:
                prompt_modifier = nn.Sequential(nn.Conv2d(chs[i], chs[i+1], kernel_size=3, stride=2, padding=(1 if i != (num_stages-1) else 0)), \
                                                nn.ReLU(inplace=True), nn.BatchNorm2d(chs[i+1]))
                mp = ModalityPromper(chs[i+1])
                
                total = sum([param.nelement() for param in mp.parameters()]) + sum([param.nelement() for param in prompt_modifier.parameters()])
                # print('P parameter: % .4fM' % (total / 1e6))

                setattr(self, f"prompt_modifier{i + 1}", prompt_modifier)
                setattr(self, f"mp{i + 1}", mp)

            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", getattr(foundation, f"patch_embed{i+1}"))
            setattr(self, f"pos_embed{i + 1}", getattr(foundation, f"pos_embed{i+1}"))
            setattr(self, f"pos_drop{i + 1}", getattr(foundation, f"pos_drop{i+1}"))
            setattr(self, f"block{i + 1}", getattr(foundation, f"block{i+1}"))


    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            print("===pretrained weight loaded===")

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x, prompt):
        outs = []

        B = x.shape[0]

        embed_l1 = getattr(self, 'embed_layer1')
        # embed_l1.eval()
        embed_l2 = getattr(self, 'embed_layer2')
        embed_l1.eval()

        embed_l2.eval()
        # print(f'before: {torch.any(torch.isnan(x))}')
        x = embed_l1(x)
        if torch.any(torch.isnan(x)):
            print(f'out 1: {torch.any(torch.isnan(x))}')
            exit()
        outs.append(x)
        x = embed_l2(x)
        if torch.any(torch.isnan(x)):

            print(f'out 2: {torch.any(torch.isnan(x))}')
            print('---------')
            exit()

        # print("--> Shape after embed {}".format(x.shape))
        if self.use_prompt:

            x, prev_prompt = self.mp0(x, self.prompt_modifier0(prompt))
            # x = x + prev_prompt

        outs.append(x)

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")

            # print('before',type(x))
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)

            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

            if self.use_prompt:
                x, prev_prompt = getattr(self, f"mp{i + 1}")(x, getattr(self, 'prompt_modifier{}'.format(i+1))(prev_prompt))
                # x = x + prev_prompt

            outs.append(x)

        return outs

    def forward(self, x, prompt, rgb_prompt=None, dep_prompt=None):
        x = self.forward_features(x, prompt)

        return x


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


class PPFTPVT(PyramidVisionTransformer):
    def __init__(self, in_chans, patch_size=4, foundation=None, **kwargs):
        super(PPFTPVT, self).__init__(
            patch_size=patch_size, in_chans=in_chans, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1, pretrained=kwargs['pretrained'], use_prompt=True, foundation=foundation)


