import timm
import torch
import torch.hub
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from torch import nn
from torch.nn import Dropout2d

default_decoder_filters = [40, 80, 128, 256]
default_recurrent_filters = [40, 80, 128, 192, 384]
default_last = 32


class BasicConvAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dilation=1, activation=nn.SiLU, bias=True):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * dilation
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=dilation,
                            bias=bias)
        self.use_act = activation is not None
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


class UnetDecoderLastConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_classes):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, num_classes, 1)
        )

    def forward(self, x):
        return self.layer(x)


class ConvBottleneck(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, dec, enc):
        x = torch.cat([dec, enc], dim=1)
        return self.seq(x)


class UnetDecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.SiLU(inplace=True)
        )

    def forward(self, x):
        return self.layer(x)


class Conv1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=None, bias=bias)


class Conv3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=None)


class ConvSilu1x1(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=1, dilation=dilation, activation=nn.SiLU)


class ConvSilu3x3(BasicConvAct):
    def __init__(self, in_channels, out_channels, dilation=1):
        super().__init__(in_channels, out_channels, kernel_size=3, dilation=dilation, activation=nn.SiLU)


class BasicUpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, activation=nn.SiLU, mode='nearest'):
        super().__init__()
        padding = int((kernel_size - 1) / 2) * 1
        self.op = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, dilation=1)
        self.use_act = activation is not None
        self.mode = mode
        if self.use_act:
            self.act = activation()

    def forward(self, x):
        x = F.upsample(x, scale_factor=2, mode=self.mode)
        x = self.op(x)
        if self.use_act:
            x = self.act(x)
        return x


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            m.weight.data = nn.init.kaiming_normal_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class TimmUnetPure(nn.Module):
    def __init__(self, encoder='tf_efficientnetv2_l_in21k',
                 in_chans=3,
                 pretrained=True,
                 channels_last=False,
                 bottleneck_type=ConvBottleneck,
                 num_classes=1, **kwargs):
        self.decoder_block = UnetDecoderBlock
        self.bottleneck_type = bottleneck_type

        backbone_arch = encoder
        img_size = kwargs.get("img_size", None)
        pretrained_img_size = kwargs.pop("pretrained_img_size", None)

        if pretrained and img_size:
            kwargs.pop("img_size")
        backbone = timm.create_model(backbone_arch, features_only=True, in_chans=in_chans, pretrained=pretrained,
                                     **kwargs)
        if pretrained and img_size and img_size != pretrained_img_size:

            state_dict = backbone.state_dict()
            del backbone
            backbone = timm.create_model(backbone_arch, features_only=True, in_chans=in_chans, pretrained=False,
                                         img_size=img_size,
                                         **kwargs)

            if "maxvit" in encoder:
                random_state = backbone.state_dict()
                new_state_dict = {}
                for k, v in state_dict.items():
                    if "relative_position_bias_table" in k:
                        print(f"recalculating {k}")
                        new_state_dict[k] = F.interpolate(v.unsqueeze(0), size=random_state[k].shape[1:],
                                                          mode="bilinear").squeeze(0)
                    else:
                        new_state_dict[k] = v
                del state_dict
                backbone.load_state_dict(new_state_dict)
                del new_state_dict
            else:
                raise NotImplementedError(f"recalculation of position table is not implement for {encoder}")

        self.filters = [f["num_chs"] for f in backbone.feature_info]

        self.decoder_filters = default_decoder_filters
        self.last_upsample_filters = default_last
        super().__init__()

        self.bottlenecks = nn.ModuleList([self.bottleneck_type(self.filters[-i - 2] + f, f) for i, f in
                                          enumerate(reversed(self.decoder_filters[:]))])

        self.decoder_stages = nn.ModuleList([self.get_decoder(idx) for idx in range(0, len(self.decoder_filters))])
        self.object_mask = UnetDecoderLastConv(self.get_last_dec_filters(), self.last_upsample_filters, num_classes)

        self.name = "u-{}".format(encoder)
        self.channels_last = channels_last
        _initialize_weights(self)
        self.dropout = Dropout2d(p=0.0)
        self.encoder = backbone

    def forward(self, x, *args, **kwargs):
        if len(x.shape) == 5:
            x = x[:, 4, ]
        if self.channels_last:
            x = x.to(memory_format=torch.channels_last)
        enc_results = []
        for enc in self.encoder(x):
            if self.channels_last:
                enc = enc.to(memory_format=torch.contiguous_format)
            enc_results.append(enc)

        x = enc_results[-1]

        bottlenecks = self.bottlenecks
        for idx, bottleneck in enumerate(bottlenecks):
            rev_idx = - (idx + 1)
            x = self.decoder_stages[rev_idx](x)
            x = bottleneck(x, enc_results[rev_idx - 1])

        mask = self.object_mask(x)
        return {"mask": mask}

    def get_last_dec_filters(self):
        return self.decoder_filters[0]

    def get_decoder(self, layer):
        in_channels = self.filters[layer + 1] if layer + 1 == len(self.decoder_filters) else self.decoder_filters[
            layer + 1]
        return self.decoder_block(in_channels, self.decoder_filters[layer], self.decoder_filters[max(layer, 0)])


if __name__ == '__main__':
    model = TimmUnetPure("maxvit_base_tf_512.in21k_ft_in1k", img_size=768).cuda().eval()
    x = torch.randn(1, 3, 768, 768).cuda()
    with torch.no_grad():
        out = model(x, )
    print(out["mask"].shape)
