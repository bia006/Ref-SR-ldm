from inspect import isfunction
import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from timm.models.layers import to_2tuple, trunc_normal_

from ldm.modules.diffusionmodules.util import checkpoint


def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class depthwise_projection(nn.Module):
    def __init__(self, 
                in_features, 
                out_features, 
                groups,
                kernel_size=(1, 1), 
                padding=(0, 0), 
                norm_type=None, 
                activation=False, 
                pointwise=False) -> None:
        super().__init__()

        self.proj = depthwise_conv_block(in_features=in_features, 
                                        out_features=out_features, 
                                        kernel_size=kernel_size,
                                        padding=padding,
                                        groups=groups,
                                        pointwise=pointwise, 
                                        norm_type=norm_type,
                                        activation=activation)
                            
    def forward(self, x):
        P = int(x.shape[1] ** 0.5)
        x = einops.rearrange(x, 'B (H W) C-> B C H W', H=P) 
        x = self.proj(x)
        x = einops.rearrange(x, 'B C H W -> B (H W) C')      
        return x

class depthwise_conv_block(nn.Module):
    def __init__(self, 
                in_features, 
                out_features,
                kernel_size=(3, 3),
                stride=(1, 1), 
                padding=(1, 1), 
                dilation=(1, 1),
                groups=None, 
                norm_type='bn',
                activation=True, 
                use_bias=True,
                pointwise=False, 
                ):
        super().__init__()
        self.pointwise = pointwise
        self.norm = norm_type
        self.act = activation
        self.depthwise = nn.Conv2d(
            in_channels=in_features,
            out_channels=in_features if pointwise else out_features,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation, 
            bias=use_bias)
        if pointwise:
            self.pointwise = nn.Conv2d(in_features, 
                                        out_features, 
                                        kernel_size=(1, 1), 
                                        stride=(1, 1), 
                                        padding=(0, 0),
                                        dilation=(1, 1), 
                                        bias=use_bias)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm2d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.depthwise(x)
        if self.pointwise:
            x = self.pointwise(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_
    # def __init__(self, in_features, out_features, n_heads=4) -> None:
    #     super().__init__()
    #     self.n_heads = n_heads

    #     self.q_map = depthwise_projection(in_features=in_features, 
    #                                         out_features=in_features, 
    #                                         groups=in_features)
    #     self.k_map = depthwise_projection(in_features=in_features, 
    #                                         out_features=in_features, 
    #                                         groups=in_features)
    #     self.v_map = depthwise_projection(in_features=out_features, 
    #                                         out_features=out_features, 
    #                                         groups=out_features)       

    #     self.projection = depthwise_projection(in_features=out_features, 
    #                                 out_features=out_features, 
    #                                 groups=out_features)                                             
    #     self.sdp = ScaleDotProduct()        

    # def forward(self, x):
    #     q, k, v = x[0], x[1], x[2]
    #     q = self.q_map(q)
    #     k = self.k_map(k)
    #     v = self.v_map(v)  
    #     b, hw, c = q.shape
    #     c_v = v.shape[2]
    #     scale = (c // self.n_heads) ** -0.5        
    #     q = q.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
    #     k = k.reshape(b, hw, self.n_heads, c // self.n_heads).permute(0, 2, 1, 3)
    #     v = v.reshape(b, hw, self.n_heads, c_v // self.n_heads).permute(0, 2, 1, 3)
    #     att = self.sdp(q, k ,v, scale).transpose(1, 2).flatten(2)    
    #     x = self.projection(att)
    #     return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.hidden_features = hidden_features
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
    
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image

    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qk_scale=None, attn_drop=0.):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
        
        self.gating_param = nn.Parameter(torch.ones(self.num_heads))

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        self.attn_drop = nn.Dropout(attn_drop)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q1, q2, k1, v1, k2):
        """
        Args:
            q: queries with shape of (num_windows*B, N, C)
            k: keys with shape of (num_windows*B, N, C)
            v: values with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = q1.shape
        q1 = q1.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k1 = k1.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v1 = v1.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        q2 = q2.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k2 = k2.reshape(B_, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        q1 = q1 * self.scale
        q2 = q2 * self.scale

        attn1 = (q1 @ k1.transpose(-2, -1))
        attn2 = (q2 @ k1.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn1 = attn1 + relative_position_bias.unsqueeze(0)
        attn2 = attn2 + relative_position_bias.unsqueeze(0)

        
        # if mask is not None:
        #     nW = mask.shape[0]
        #     attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
        #     attn = attn.view(-1, self.num_heads, N, N)
        #     attn = self.softmax(attn)
        # else:
        attn1 = self.softmax(attn1)
        attn2 = self.softmax(attn2)

        gating = self.gating_param.view(1,-1,1,1)
        attn = (1.-torch.sigmoid(gating)) * attn1 + torch.sigmoid(gating) * attn2
        attn /= attn.sum(dim=-1).unsqueeze(-1)

        x = (attn @ v1).transpose(1, 2).reshape(B_, N, C)
        
        return x

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'

    def flops(self, N):
        # calculate flops for 1 window with token length of N
        flops = 0
        # qkv = self.qkv(x)
        flops += N * self.dim * 3 * self.dim
        # attn = (q @ k.transpose(-2, -1))
        flops += self.num_heads * N * (self.dim // self.num_heads) * N
        #  x = (attn @ v)
        flops += self.num_heads * N * N * (self.dim // self.num_heads)
        # x = self.proj(x)
        flops += N * self.dim * self.dim
        return flops


class CrossAttention(nn.Module):
    r""" StyleSwin Transformer Block.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        context_dim (int): Dimension of style vector.
    """

    def __init__(self, dim, input_resolution, num_heads, context_dim=None, window_size=8,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 act_layer=nn.GELU):
        super().__init__()
        self.dim = dim
        input_resolution=to_2tuple(input_resolution)
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.shift_size = self.window_size // 2
        self.context_dim = context_dim

        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.attn = nn.ModuleList([
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qk_scale=qk_scale, attn_drop=attn_drop),
            WindowAttention(
                dim // 2, window_size=to_2tuple(self.window_size), num_heads=num_heads,
                qk_scale=qk_scale, attn_drop=attn_drop),
        ])
        
        # attn_mask1 = None
        # attn_mask2 = None
        # if self.shift_size > 0:
        #     # calculate attention mask for SW-MSA
        #     H, W = self.input_resolution
        #     img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
        #     h_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     w_slices = (slice(0, -self.window_size),
        #                 slice(-self.window_size, -self.shift_size),
        #                 slice(-self.shift_size, None))
        #     cnt = 0
        #     for h in h_slices:
        #         for w in w_slices:
        #             img_mask[:, h, w, :] = cnt
        #             cnt += 1

        #     # nW, window_size, window_size, 1
        #     mask_windows = window_partition(img_mask, self.window_size)
        #     mask_windows = mask_windows.view(-1,
        #                                     self.window_size * self.window_size)
        #     attn_mask2 = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        #     attn_mask2 = attn_mask2.masked_fill(
        #         attn_mask2 != 0, float(-100.0)).masked_fill(attn_mask2 == 0, float(0.0))
        
        # self.register_buffer("attn_mask1", attn_mask1)
        # self.register_buffer("attn_mask2", attn_mask2)

        self.norm1 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, context):
        b, c, h, w = context.shape
        H, W = self.input_resolution
        B, L, C = x.shape

        if L == H * W:       
            # x_reshape = x.reshape(B, C, H, W)
            # context = F.interpolate(context, size=x_reshape.shape[2:], mode='bilinear', align_corners=False)
            context = context.reshape(B, H * W, C)
            H, W = H, W
        elif L == H//2 * W//2:
            x_reshape = x.reshape(B, C, H//2, W//2)
            context = F.interpolate(context, size=x_reshape.shape[2:], mode='bicubic', align_corners=False)
            context = context.reshape(B, H//2 * W//2, C)
            H, W = H//2, W//2
        elif L == H//4 * W//4:
            x_reshape = x.reshape(B, C, H//4, W//4)
            context = F.interpolate(context, size=x_reshape.shape[2:], mode='bicubic', align_corners=False)
            context = context.reshape(B, H//4 * W//4, C)
            H, W = H//4, W//4
        elif L == H//5 * W//5:
            x_reshape = x.reshape(B, C, H//5, W//5)
            context = F.interpolate(context, size=x_reshape.shape[2:], mode='bicubic', align_corners=False)
            context = context.reshape(B, H//5 * W//5, C)
            H, W = H//5, W//5

        # Double Attn
        shortcut = x
        # x = self.norm1(x.transpose(-1, -2), style).transpose(-1, -2)
        
        qkv_1 = self.qkv(x).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_2 = self.qkv(context).reshape(B, -1, 3, C).permute(2, 0, 1, 3).reshape(3 * B, H, W, C)
        qkv_1 = qkv_1[:, :, :, : C // 2].reshape(3, B, H, W, C // 2)
        if self.shift_size > 0:
            qkv_2 = torch.roll(qkv_2[:, :, :, C // 2:], shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)).reshape(3, B, H, W, C // 2)
        else:
            qkv_2 = qkv_2[:, :, :, C // 2:].reshape(3, B, H, W, C // 2)
        
        q1_windows, k1_windows, v1_windows = self.get_window_qkv(qkv_1)
        q2_windows, k2_windows, v2_windows = self.get_window_qkv(qkv_2)

        x1 = self.attn[0](q1_windows, q2_windows, k1_windows, v2_windows, k2_windows)
        x2 = self.attn[1](q1_windows, q2_windows, k1_windows, v2_windows, k2_windows)
        
        x1 = window_reverse(x1.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)
        x2 = window_reverse(x2.view(-1, self.window_size * self.window_size, C // 2), self.window_size, H, W)

        if self.shift_size > 0:
            x2 = torch.roll(x2, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x2 = x2

        x = torch.cat([x1.reshape(B, H * W, C // 2), x2.reshape(B, H * W, C // 2)], dim=2)
        x = self.proj(x)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm1(x))

        return x
    
    def get_window_qkv(self, qkv):
        q, k, v = qkv[0], qkv[1], qkv[2]   # B, H, W, C
        C = q.shape[-1]
        q_windows = window_partition(q, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        k_windows = window_partition(k, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        v_windows = window_partition(v, self.window_size).view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        return q_windows, k_windows, v_windows

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += 1 * self.context_dim * self.dim * 2
        flops += 2 * (H * W) * self.dim
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        for attn in self.attn:
            flops += nW * (attn.flops(self.window_size * self.window_size))
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += 1 * self.context_dim * self.dim * 2
        flops += 2 * (H * W) * self.dim
        return flops
    

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, depth, num_heads, input_resolution, window_size=10, out_dim=None,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, dropout=0., attn_drop=0., upsample=None, 
                 use_checkpoint=False, context_dim=520, gated_ff=True, checkpoint=True):
        super().__init__()
        self.input_resolution = input_resolution
        # self.attn1 = nn.ModuleList([CrossAttention(in_features=sum(dim),
        #                                         out_features=dim,
        #                                         n_heads=n_heads, 
        #                                 ) for feature, head in zip(dim, channel_head)])  # is a self-attention
        # self.attn2 = nn.ModuleList([SpatialAttention(
        #                                             in_features=sum(dim),
        #                                             out_features=dim,
        #                                             n_heads=n_heads, 
        #                                             ) 
        #                                             for feature, head in zip(dim, spatial_head)])
        self.attn1 = CrossAttention(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=dropout, attn_drop=attn_drop, context_dim=context_dim)  # is self-attn if context is none
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(dim=dim, input_resolution=input_resolution,
                                    num_heads=num_heads, window_size=window_size,
                                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                    drop=dropout, attn_drop=attn_drop, context_dim=context_dim)  # is self-attn if context is none
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        self.checkpoint = checkpoint

    def forward(self, x, context=None):
        return checkpoint(self._forward, (x, context), self.parameters(), self.checkpoint)

    def _forward(self, x, context=None):
        H, W = self.input_resolution, self.input_resolution
        B, L, C = x.shape
        # assert L == H * W, "input feature has wrong size"
        x = self.attn1(self.norm1(x), context=context) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        # else: 
        #     x = self.attn3(self.norm1(x), context=context) + x
        #     x = self.attn4(self.norm2(x), context=context) + x
        #     x = self.ff(self.norm3(x)) + x
        return x


class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, num_heads, d_head, input_resolution,
                 depth=1, dropout=0., window_size=8, context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = num_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, num_heads, d_head, input_resolution, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in