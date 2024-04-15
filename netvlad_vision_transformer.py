from base_model import BaseModel
import warnings
import math
import torch
from pathlib import Path
import logging
import subprocess
from scipy.io import loadmat
from torch import nn
import torch.nn.functional as F
import numpy as np

logger = logging.getLogger(__name__)
EPS = 1e-6

def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


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
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        y, attn = self.attn(self.norm1(x))
        if return_attention:
            return attn
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        num_patches = (img_size // patch_size) * (img_size // patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=[224], patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, **kwargs):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(
            img_size=img_size[0], patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        # self.head = nn.Sequential(*[nn.Linear(2 * embed_dim, embed_dim), nn.GELU(),
        #                             nn.Linear(embed_dim, num_classes)]) if num_classes > 0 else nn.Identity()
        self.head = nn.Sequential(*[nn.Linear(embed_dim, embed_dim), nn.GELU(),
                                    nn.Linear(embed_dim, num_classes)]) if num_classes > 0 else nn.Identity()
        
        self.top = nn.Conv1d(in_channels=197, out_channels=512, kernel_size=1, bias=False)

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        class_pos_embed = self.pos_embed[:, 0]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_embed.patch_size
        h0 = h // self.patch_embed.patch_size
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)

    def prepare_tokens(self, x):
        B, nc, w, h = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.interpolate_pos_encoding(x, w, h)

        return self.pos_drop(x)

    def forward(self, x, classify=False):
        x = self.prepare_tokens(x)
        for blk in self.blocks:
            x = blk(x)

        x = self.norm(x)

        b, c, w = x.size()
        
        output_tensor = self.top(x)
        # output_tensor = self.head(x)
        x = output_tensor
        return x

class NetVLADLayer(nn.Module):
    def __init__(self, input_dim=512, K=64, score_bias=False, intranorm=True):
        super().__init__()
        self.score_proj = nn.Conv1d(
            input_dim, K, kernel_size=1, bias=score_bias)
        centers = nn.parameter.Parameter(torch.empty([input_dim, K]))
        nn.init.xavier_uniform_(centers)
        self.register_parameter('centers', centers)
        self.intranorm = intranorm
        self.output_dim = input_dim * K

    def forward(self, x):
        b = x.size(0)
        scores = self.score_proj(x)
        scores = F.softmax(scores, dim=1)
        diff = (x.unsqueeze(2) - self.centers.unsqueeze(0).unsqueeze(-1))
        desc = (scores.unsqueeze(1) * diff).sum(dim=-1)
        if self.intranorm:
            # From the official MATLAB implementation.
            desc = F.normalize(desc, dim=1)
        desc = desc.view(b, -1)
        desc = F.normalize(desc, dim=1)
        return desc

class NetVLAD(BaseModel):
    default_conf = {
        'model_name': 'VGG16-NetVLAD-Pitts30K',
        'whiten': True
    }
    required_inputs = ['image']

    # Models exported using
    # https://github.com/uzh-rpg/netvlad_tf_open/blob/master/matlab/net_class2struct.m.
    dir_models = {
        'VGG16-NetVLAD-Pitts30K': 'https://cvg-data.inf.ethz.ch/hloc/netvlad/Pitts30K_struct.mat',
        'VGG16-NetVLAD-TokyoTM': 'https://cvg-data.inf.ethz.ch/hloc/netvlad/TokyoTM_struct.mat'
    }

    def _init(self, conf):
        assert conf['model_name'] in self.dir_models.keys()

        # Download the checkpoint.
        checkpoint = Path(torch.hub.get_dir(), 'netvlad', 'Pitts30K_struct' + '.mat')
        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True, parents=True)
            link = self.dir_models[conf['model_name']]
            cmd = ['wget', link, '-O', str(checkpoint)]
            logger.info(f'Downloading the NetVLAD model with `{cmd}`.')
            subprocess.run(cmd, check=True)

        # Create the network.
        # Remove classification head.
        #backbone = list(models.vgg16().children())[0]
        # Remove last ReLU + MaxPool2d.
        #self.backbone = nn.Sequential(*list(backbone.children())[: -2])
        self.backbone = VisionTransformer(img_size=[224], patch_size=16, in_chans=3, num_classes=512, embed_dim=768,
                                       depth=12,
                                       num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0.,
                                       attn_drop_rate=0.,
                                       drop_path_rate=0., norm_layer=nn.LayerNorm)

        self.netvlad = NetVLADLayer()

        if conf['whiten']:
            self.whiten = nn.Linear(self.netvlad.output_dim, 4096)

        # Parse MATLAB weights using https://github.com/uzh-rpg/netvlad_tf_open
        mat = loadmat(checkpoint, struct_as_record=False, squeeze_me=True)

        # CNN weights.
        for layer, mat_layer in zip(self.backbone.children(),
                                    mat['net'].layers):
            if isinstance(layer, nn.Conv2d):
                w = mat_layer.weights[0]  # Shape: S x S x IN x OUT
                b = mat_layer.weights[1]  # Shape: OUT
                # Prepare for PyTorch - enforce float32 and right shape.
                # w should have shape: OUT x IN x S x S
                # b should have shape: OUT
                w = torch.tensor(w).float().permute([3, 2, 0, 1])
                b = torch.tensor(b).float()
                # Update layer weights.
                layer.weight = nn.Parameter(w)
                layer.bias = nn.Parameter(b)

        # NetVLAD weights.
        score_w = mat['net'].layers[30].weights[0]  # D x K
        # centers are stored as opposite in official MATLAB code
        center_w = -mat['net'].layers[30].weights[1]  # D x K
        # Prepare for PyTorch - make sure it is float32 and has right shape.
        # score_w should have shape K x D x 1
        # center_w should have shape D x K
        score_w = torch.tensor(score_w).float().permute([1, 0]).unsqueeze(-1)
        center_w = torch.tensor(center_w).float()
        # Update layer weights.
        self.netvlad.score_proj.weight = nn.Parameter(score_w)
        self.netvlad.centers = nn.Parameter(center_w)        
        self.linearT = nn.Linear(197*768, 512*3)

        # Whitening weights.
        if conf['whiten']:
            w = mat['net'].layers[33].weights[0]  # Shape: 1 x 1 x IN x OUT
            b = mat['net'].layers[33].weights[1]  # Shape: OUT
            # Prepare for PyTorch - make sure it is float32 and has right shape
            w = torch.tensor(w).float().squeeze().permute([1, 0])  # OUT x IN
            b = torch.tensor(b.squeeze()).float()  # Shape: OUT
            # Update layer weights.
            self.whiten.weight = nn.Parameter(w)
            self.whiten.bias = nn.Parameter(b)

        # Preprocessing parameters.
        self.preprocess = {
            'mean': mat['net'].meta.normalization.averageImage[0, 0],
            'std': np.array([1, 1, 1], dtype=np.float32)
        }

    def _forward(self, image):
        assert image.shape[1] == 3        
        mean = self.preprocess['mean']
        std = self.preprocess['std']
        image = image - image.new_tensor(mean).view(1, -1, 1, 1)
        image = image / image.new_tensor(std).view(1, -1, 1, 1)

        # Feature extraction.
        descriptors = self.backbone(image)        
        b, c, w = descriptors.size()        
        # desc_reshaped = descriptors.view(b,-1)        
        # output_tensor = self.linearT(desc_reshaped)
        # output_tensor = output_tensor.view(b, 512, 3)
        # descriptors = output_tensor

        conv_layer = nn.Conv1d(in_channels=c, out_channels=512, kernel_size=1, bias=False)
        output_tensor = conv_layer(descriptors)
        descriptors = output_tensor

        # NetVLAD layer.
        descriptors = F.normalize(descriptors, dim=1)  # Pre-normalization.
        desc = self.netvlad(descriptors)

        # Whiten if needed.
        if hasattr(self, 'whiten'):
            desc = self.whiten(desc)
            desc = F.normalize(desc, dim=1)  # Final L2 normalization.

        return desc
        
device = torch.device("cuda")
default_conf = {
        'model_name': 'VGG16-NetVLAD-Pitts30K',
        'whiten': True
    }
model = NetVLAD(default_conf)
model.to(device)
#x = torch.randn(1, 3, 224, 224)
x = torch.rand(1, 3, 224, 224)*255
#x = torch.rand(1, 3, 448, 448)*255
x = x.to(torch.uint8)
x = x.cuda()
output = model(x)
print(output.shape)