import torch
from einops import rearrange
from einops.layers.torch import Rearrange
from torch import nn, einsum
from torch.nn import functional as F
import math
from Nets.SwinTransformer import SwinTransformerBlock2d
from thop import profile, clever_format


# MSFE
class Multi_Scale_Feature_Extract_Module(nn.Module):
    def __init__(self):
        super().__init__()
        self.Initial = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_1 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1, stride=1, dilation=1),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_2 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=2, stride=1, dilation=2),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_3 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=4, stride=1, dilation=4),
            nn.BatchNorm2d(16, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.dilatation_conv_4 = nn.Sequential(
            nn.Conv2d(16 * 3, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.Initial(x)
        x1 = self.dilatation_conv_1(x)
        x2 = self.dilatation_conv_2(x)
        x3 = self.dilatation_conv_3(x)
        concatenation = torch.cat([x1, x2, x3], dim=1)
        x4 = self.dilatation_conv_4(concatenation)
        x = x4 + residual
        x = self.relu(x)
        return x

def get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


def make_cbr(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.PReLU())


def make_cbg(in_dim, out_dim):
    return nn.Sequential(nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1), nn.BatchNorm2d(out_dim), nn.GELU())


def rescale_to(x, scale_factor: float = 2, interpolation='nearest'):
    return F.interpolate(x, scale_factor=scale_factor, mode=interpolation)


def resize_as(x, y, interpolation='bilinear'):
    return F.interpolate(x, size=y.shape[-2:], mode=interpolation)


def image2patches(x):
    """b c (hg h) (wg w) -> (hg wg b) c h w"""
    x = rearrange(x, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
    return x


def patches2image(x):
    """(hg wg b) c h w -> b c (hg h) (wg w)"""
    x = rearrange(x, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)
    return x

class PositionEmbeddingSine:
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale
        self.dim_t = torch.arange(0, self.num_pos_feats, dtype=torch.float32, device='cuda')

    def __call__(self, b, h, w):
        mask = torch.zeros([b, h, w], dtype=torch.bool, device='cuda')
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(dim=1, dtype=torch.float32)
        x_embed = not_mask.cumsum(dim=2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = ((y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * self.scale).cuda()
            x_embed = ((x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * self.scale).cuda()

        dim_t = self.temperature ** (2 * (self.dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(
            3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(
            3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

class MVL(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[1, 2, 4]):
        super(MVL, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.linear5 = nn.Linear(d_model, d_model * 2)
        self.linear6 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.activation = get_activation_fn('relu')
        self.pool_ratios = pool_ratios
        self.p_poses = []
        self.g_pos = None
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)

    def forward(self, l, g):
        """
        l: 4,c,h,w
        g: 1,c,h,w
        """
        b, c, h, w = l.size()
        # 4,c,h,w -> 1,c,2h,2w
        concated_locs = rearrange(l, '(hg wg b) c h w -> b c (hg h) (wg w)', hg=2, wg=2)

        pools = []
        for pool_ratio in self.pool_ratios:
            # b,c,h,w
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(concated_locs, tgt_hw)
            pools.append(rearrange(pool, 'b c h w -> (h w) b c'))
            if self.g_pos is None:
                pos_emb = self.positional_encoding(pool.shape[0], pool.shape[2], pool.shape[3])
                pos_emb = rearrange(pos_emb, 'b c h w -> (h w) b c')
                self.p_poses.append(pos_emb)
        pools = torch.cat(pools, 0)
        if self.g_pos is None:
            self.p_poses = torch.cat(self.p_poses, dim=0)
            pos_emb = self.positional_encoding(g.shape[0], g.shape[2], g.shape[3])
            self.g_pos = rearrange(pos_emb, 'b c h w -> (h w) b c')

        # attention between glb (q) & multisensory concated-locs (k,v)
        g_hw_b_c = rearrange(g, 'b c h w -> (h w) b c')
        g_hw_b_c = g_hw_b_c + self.dropout1(self.attention[0](g_hw_b_c + self.g_pos, pools + self.p_poses, pools)[0])
        g_hw_b_c = self.norm1(g_hw_b_c)
        g_hw_b_c = g_hw_b_c + self.dropout2(self.linear6(self.dropout(self.activation(self.linear5(g_hw_b_c)).clone())))
        g_hw_b_c = self.norm2(g_hw_b_c)


        # attention between origin locs (q) & freashed glb (k,v)
        l_hw_b_c = rearrange(l, "b c h w -> (h w) b c")
        _g_hw_b_c = rearrange(g_hw_b_c, '(h w) b c -> h w b c', h=h, w=w)
        _g_hw_b_c = rearrange(_g_hw_b_c, "(ng h) (nw w) b c -> (h w) (ng nw b) c", ng=2, nw=2)
        # print(_g_hw_b_c.shape)
        outputs_re = []
        for i, (_l, _g) in enumerate(zip(l_hw_b_c.chunk(4, dim=1), _g_hw_b_c.chunk(4, dim=1))):
            outputs_re.append(self.attention[i + 1](_l, _g, _g)[0])  # (h w) 1 c
        outputs_re = torch.cat(outputs_re, 1)  # (h w) 4 c

        l_hw_b_c = l_hw_b_c + self.dropout1(outputs_re)
        l_hw_b_c = self.norm1(l_hw_b_c)
        l_hw_b_c = l_hw_b_c + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(l_hw_b_c)).clone())))
        l_hw_b_c = self.norm2(l_hw_b_c)

        l = torch.cat((l_hw_b_c, g_hw_b_c), 1)  # hw,b(5),c
        return rearrange(l, "(h w) b c -> b c h w", h=h, w=w)  ## (5,c,h*w)

class MVC(nn.Module):
    def __init__(self, d_model, num_heads, pool_ratios=[4, 8, 16], h=None):
        super(MVC, self).__init__()
        self.attention = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1),
            nn.MultiheadAttention(d_model, num_heads, dropout=0.1)
        ])

        self.linear3 = nn.Linear(d_model, d_model * 2)
        self.linear4 = nn.Linear(d_model * 2, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.sigmoid = nn.Sigmoid()
        self.activation = get_activation_fn('relu')
        self.sal_conv = nn.Conv2d(d_model, 1, 1)
        self.pool_ratios = pool_ratios
        self.positional_encoding = PositionEmbeddingSine(num_pos_feats=d_model // 2, normalize=True)
    def forward(self, x):
        b, c, h, w = x.size()
        loc, glb = x.split([4, 1], dim=0)  # 4,c,h,w; 1,c,h,w
        # b(4),c,h,w
        patched_glb = rearrange(glb, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)

        # generate token attention map
        token_attention_map = self.sigmoid(self.sal_conv(glb))
        token_attention_map = F.interpolate(token_attention_map, size=patches2image(loc).shape[-2:], mode='nearest')
        loc = loc * rearrange(token_attention_map, 'b c (hg h) (wg w) -> (hg wg b) c h w', hg=2, wg=2)
        pools = []
        for pool_ratio in self.pool_ratios:
            tgt_hw = (round(h / pool_ratio), round(w / pool_ratio))
            pool = F.adaptive_avg_pool2d(patched_glb, tgt_hw)
            pools.append(rearrange(pool, 'nl c h w -> nl c (h w)'))  # nl(4),c,hw
        # nl(4),c,nphw -> nl(4),nphw,1,c
        pools = rearrange(torch.cat(pools, 2), "nl c nphw -> nl nphw 1 c")
        loc_ = rearrange(loc, 'nl c h w -> nl (h w) 1 c')
        outputs = []
        for i, q in enumerate(loc_.unbind(dim=0)):  # traverse all local patches
        # np*hw,1,c
            v = pools[i]
            k = v
            outputs.append(self.attention[i](q, k, v)[0])
        outputs = torch.cat(outputs, 1)
        src = loc.view(4, c, -1).permute(2, 0, 1) + self.dropout1(outputs)
        src = self.norm1(src)
        src = src + self.dropout2(self.linear4(self.dropout(self.activation(self.linear3(src)).clone())))
        src = self.norm2(src)

        src = src.permute(1, 2, 0).reshape(4, c, h, w)  # freshed loc
        glb = glb + F.interpolate(patches2image(src), size=glb.shape[-2:], mode='nearest')  # freshed glb
        return torch.cat((src, glb), 0)


class MVF(nn.Module):
    def __init__(self, emb_dim=32):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv2d(emb_dim, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Conv2d(64, emb_dim, kernel_size=3, padding=1)
        )

        self.br = nn.Sequential(
            nn.Conv2d(32, 48, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(48, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, loc, glb):
        output_cat = loc + resize_as(glb, loc)
        x = self.conv_head(output_cat)
        residual = x
        x = self.br(x)
        x = x + residual
        x = self.relu(x)
        return x


class MVPFusion(nn.Module):
    def __init__(
            self,
            *,
            img_channels=3,
            dropout=0.
    ):
        super().__init__()

        # Shared feature extraction
        self.sfe = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32, eps=1e-5, momentum=0.1),
            nn.ReLU(inplace=True),
        )

        self.dfe = nn.Sequential(
            SwinTransformerBlock2d(32, (16, 16), 1),
            SwinTransformerBlock2d(32, (16, 16), 2),
            SwinTransformerBlock2d(32, (16, 16), 4),
            SwinTransformerBlock2d(32, (16, 16), 8),
        )



        emb_dim=32
        self.mvl = MVL(emb_dim, 1, [1, 2, 4])
        self.mvc1 = MVC(emb_dim, 1, [2, 4, 8])
        self.mvc2 = MVC(emb_dim, 1, [2, 4, 8])
        self.mvf = MVF(emb_dim)


        self.sideout1 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.sideout2 = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))
        self.conv1 = make_cbr(emb_dim, emb_dim)
        self.output = nn.Sequential(nn.Conv2d(emb_dim, 1, kernel_size=3, padding=1))

    def forward(self, A, B):
        # Shallow feature extraction
        A = self.sfe(A)
        B = self.sfe(B)

        # Deep feature extraction
        Feature_A = self.dfe(A)
        Feature_B = self.dfe(B)

        glb = rescale_to(Feature_B, scale_factor=0.5, interpolation='bilinear')
        loc = image2patches(Feature_A)

        # Multi-view localization
        concatenation = self.mvl(loc, glb)

        # Multi-view complementary
        concatenation1 = self.mvc1(concatenation)
        concatenation2 = self.mvc2(concatenation1)

        concatenation2 = self.conv1(concatenation2)
        loc, glb = concatenation2.split([4, 1], dim=0)
        output_cat = patches2image(loc)

        # Multi-view fusion
        final_output = self.mvf(output_cat, glb)

        final_output = self.output(final_output)

        # Calculate the intermediate results
        sideout2 = self.sideout2(concatenation2).cuda()
        glb2 = sideout2[-1, :, :, :].unsqueeze(0)
        sideout2 = patches2image(sideout2[:-1]).cuda()

        sideout1 = self.sideout1(concatenation1).cuda()
        glb1 = sideout1[-1, :, :, :].unsqueeze(0)
        sideout1 = patches2image(sideout1[:-1]).cuda()

        return final_output, sideout2, glb2, sideout1, glb1


if __name__ == '__main__':
    test_tensor_A = torch.zeros((1, 3, 112, 112)).to('cuda')
    test_tensor_B = torch.rand((1, 3, 112, 112)).to('cuda')
    model = MVPFusion().to('cuda')
    num_params = 0
    flops, params = profile(model, inputs=(test_tensor_A, test_tensor_B))
    flops, params = clever_format([flops, params], "%.6f")

    print(f"🔥 模型 FLOPs: {flops}")
    print(f"🔥 模型参数量: {params}")
    for p in model.parameters():
        num_params += p.numel()
    # print(model)
    # print("The number of model parameters: {} M\n\n".format(round(num_params / 10e5, 6)))
    # final_output,sideout2,glb2,sideout1,glb1 = model(test_tensor_A, test_tensor_B)
    # print(final_output.shape)
    # print(sideout2.shape)
    # print(glb2.shape)
