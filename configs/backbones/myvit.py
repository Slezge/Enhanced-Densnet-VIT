import torch
from torch import nn, einsum
from utils.drop_path import DropPath
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from configs.backbones.STP import ShiftedPatchTokenization
import torch.nn.functional as F
import importlib
from ..common import BaseModule
from .densenet import  DenseNet

# helpers
from .resnet import ResNet
class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x


def init_weights(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.LayerNorm):
        nn.init.constant_(m.bias, 0)
        nn.init.constant_(m.weight, 1.0)


class PreNorm(nn.Module):
    def __init__(self, num_tokens, dim, fn):
        super().__init__()
        self.dim = dim
        self.num_tokens = num_tokens
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, num_patches, hidden_dim, dropout=0.):
        super().__init__()
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.num_patches = num_patches

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, num_patches, heads=8, dim_head=64, dropout=0., is_LSA=False):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)
        self.num_patches = num_patches
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.dim = dim
        self.inner_dim = inner_dim
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(self.dim, self.inner_dim * 3, bias=False)
        init_weights(self.to_qkv)
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, self.dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

        if is_LSA:
            self.scale = nn.Parameter(self.scale * torch.ones(heads))
            self.mask = torch.eye(self.num_patches + 1, self.num_patches + 1)
            self.mask = torch.nonzero((self.mask == 1), as_tuple=False)
        else:
            self.mask = None

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        if self.mask is None:
            dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        else:
            scale = self.scale
            dots = torch.mul(einsum('b h i d, b h j d -> b h i j', q, k),
                             scale.unsqueeze(0).unsqueeze(-1).unsqueeze(-1).expand((b, h, 1, 1)))
            dots[:, :, self.mask[:, 0], self.mask[:, 1]] = -987654321

        attn = self.attend(dots)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

    def flops(self):
        flops = 0
        if not self.is_coord:
            flops += self.dim * self.inner_dim * 3 * (self.num_patches + 1)
        else:
            flops += (self.dim + 2) * self.inner_dim * 3 * self.num_patches
            flops += self.dim * self.inner_dim * 3


class Transformer(nn.Module):
    def __init__(self, dim, num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout=0., stochastic_depth=0.,
                 is_LSA=False):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.scale = {}

        for i in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(num_patches, dim,
                        Attention(dim, num_patches, heads=heads, dim_head=dim_head, dropout=dropout, is_LSA=is_LSA)),
                PreNorm(num_patches, dim, FeedForward(dim, num_patches, dim * mlp_dim_ratio, dropout=dropout))
            ]))
        self.drop_path = DropPath(stochastic_depth) if stochastic_depth > 0 else nn.Identity()

    def forward(self, x):
        for i, (attn, ff) in enumerate(self.layers):
            x = self.drop_path(attn(x)) + x
            x = self.drop_path(ff(x)) + x
            self.scale[str(i)] = attn.fn.scale
        return x


class MYViT(nn.Module):
    def __init__(self,img_size=224,patch_size=16, dim=1024, depth=12, heads=12, mlp_dim_ratio=2, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., stochastic_depth=0., is_LSA=True, is_SPT=False):
        super(MYViT,self).__init__()
        self.conv1=nn.Conv2d(1024,1024,kernel_size=1,stride=1)
        self.bn1=nn.BatchNorm2d(1024)
        #self.conv4 = nn.Conv2d(2048, 1024, kernel_size=1, stride=1)
        #self.bn4 = nn.BatchNorm2d(1024)
        self.conv2=nn.Conv2d(1024,512,kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.CBAM=CBAMLayer(channel=512)
        self.conv3=nn.Conv2d(512,1024,kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.gap=nn.AvgPool2d(kernel_size=(7,7))
        self.fc=nn.Linear(1024,5)
        self.relu=nn.ReLU(True)
        self.densenet=DenseNet(arch='121')



        #image_height, image_width = pair(img_size)
        #patch_height, patch_width = pair(patch_size)
        #self.num_patches = (image_height // patch_height) * (image_width // patch_width)
        self.patch_dim = 1024
        self.num_patches = 49
        self.dim = dim
        #self.resnet= ResNet(depth=18, num_stages=4, out_indices=(3,))

        #if not is_SPT:
        self.to_patch_embedding = nn.Sequential(
               #Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = 16, p2 = 16),
                Rearrange('b c h w -> b (h w)  c'),
                nn.Linear(self.patch_dim, self.dim)
            )

        #else:
            #self.to_patch_embedding = ShiftedPatchTokenization(3, self.dim, patch_size, is_pe=True)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, self.dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, self.dim))
        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(self.dim, self.num_patches, depth, heads, dim_head, mlp_dim_ratio, dropout,
                                       stochastic_depth, is_LSA=is_LSA)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(self.dim),

        )

        self.apply(init_weights)

    def forward(self, img):
        # patch embedding
        net_dict = self.densenet.state_dict()
        predict_model = torch.load('configs/densenet121_4xb256_in1k_20220426-07450f99.pth')
        state_dict = {k: v for k, v in predict_model.items() if k in net_dict.keys()}
        net_dict.update(state_dict)
        self.densenet.load_state_dict(net_dict)
        x=self.densenet(img)
        #x=self.resnet(img)
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.relu(x)
        #x = self.conv4(x)
        #x = self.bn4(x)
        #x = self.relu(x)
        y=x
        y=self.conv2(x)
        y=self.bn2(y)
        y=self.relu(y)
        y=self.CBAM(y)
        y=self.conv3(y)
        y=self.bn3(y)
        y = self.relu(y)
        y=y+x
        y=self.relu(y)
        y=self.gap(y)
        y1=torch.squeeze(y)
        y=rearrange(y,'b c h w -> (b h)  (c w)')




        x = self.to_patch_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)
        end=torch.cat([self.mlp_head(x[:, 0]), y1],dim=1)


        return end