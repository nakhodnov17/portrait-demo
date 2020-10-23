import torch

import pretrainedmodels


# noinspection PyAbstractClass
class Identity(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return x


# noinspection PyAbstractClass
class OneToMultiBranch(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()

        self.transforms = torch.nn.ModuleList(modules)

    def forward(self, x):
        result = []
        for module in self.transforms:
            result.append(module(x))

        return result


# noinspection PyAbstractClass
class MultiToMultiBranch(torch.nn.Module):
    def __init__(self, modules):
        super().__init__()

        self.transforms = torch.nn.ModuleList(modules)

    def forward(self, xs):
        result = []
        for x, module in zip(xs, self.transforms):
            result.append(module(x))

        return result


# noinspection PyAbstractClass
class CCPB(torch.nn.Module):
    def __init__(self, C, C_prime, C_0, C_1, C_2, L):
        super().__init__()

        assert C_prime == C_0 + C_1 + C_2

        (
            self.C, self.C_prime, self.C_0, self.C_1, self.C_2, self.L
        ) = (
            C, C_prime, C_0, C_1, C_2, L
        )

        self.conv_prime = torch.nn.Conv2d(C, C_prime, kernel_size=(1, 1))

        self.branch_0 = torch.nn.Sequential(
            torch.nn.Conv2d(C_0, C_0, kernel_size=(1, 1))
        )

        self.branch_1 = torch.nn.Sequential(
            torch.nn.Conv2d(C_1, C_1, kernel_size=(1, 1)),
            torch.nn.Conv2d(C_1, C_1, kernel_size=(3, 3), padding=1)
        )

        self.branch_2 = torch.nn.Sequential(
            torch.nn.Conv2d(C_2, C_2, kernel_size=(1, 1)),
            torch.nn.Conv2d(C_2, C_2, kernel_size=(3, 3), padding=1),
            torch.nn.Conv2d(C_2, C_2, kernel_size=(3, 3), padding=1)
        )

        self.conv_out = torch.nn.Conv2d(C_prime, L, kernel_size=(1, 1))

    def forward(self, x):
        x = self.conv_prime(x)

        x_cat = torch.cat([
            self.branch_0(x[:, :self.C_0]),
            self.branch_1(x[:, self.C_0:self.C_0 + self.C_1]),
            self.branch_2(x[:, self.C_0 + self.C_1:])
        ], dim=1)

        x = self.conv_out(x + x_cat)

        return x


# noinspection PyAbstractClass
class AdaptiveConvolution(torch.nn.Module):
    def __init__(self, ccpb_parameters):
        super().__init__()

        self.ccpb_parameters = ccpb_parameters
        self.adaptive_conv = MultiToMultiBranch([
            CCPB(*ccpb_parameter) for ccpb_parameter in ccpb_parameters
        ])

    def forward(self, xs):
        return self.adaptive_conv(xs)


# noinspection PyAbstractClass
class ChannelMaxPool(torch.nn.Module):
    def __init__(self):
        super().__init__()

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return torch.max(x, dim=1, keepdim=True)[0]


# noinspection PyAbstractClass
class MCFB(torch.nn.Module):
    def __init__(self, L, K, H, W):
        super().__init__()

        assert K % 2 == 1

        self.L, self.K, self.H, self.W = L, K, H, W

        self.mcfb_branches = OneToMultiBranch([
            torch.nn.Sequential(
                torch.nn.Conv2d(4 * L, L, kernel_size=(3, 3), padding=1)
            ),
            torch.nn.Sequential(
                torch.nn.Conv2d(4 * L, L, kernel_size=(1, K), padding=(0, (K - 1) // 2)),
                torch.nn.Conv2d(L, L, kernel_size=(K, 1), padding=((K - 1) // 2, 0)),
            ),
            torch.nn.Sequential(
                ChannelMaxPool(),
                torch.nn.Conv2d(1, 1, kernel_size=(1, W)),
                torch.nn.Conv2d(1, 1, kernel_size=(H, 1)),
                torch.nn.Sigmoid()
            )
        ])

    def forward(self, x):
        branches = self.mcfb_branches(x)

        return (branches[0] + branches[1]) * (1.0 + branches[2])


# noinspection PyAbstractClass
class MultiResolutionFusion(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.mr_fusion = MultiToMultiBranch([
            Identity(),
            torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
            torch.nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
        ])

    def forward(self, xs):
        assert len(xs) == 4

        return torch.cat(self.mr_fusion(xs), dim=1)


# noinspection PyAbstractClass
class BCIB(torch.nn.Module):
    def __init__(self, C, K):
        super().__init__()

        assert K % 2 == 1

        self.C = C
        self.K = K

        self.bcib_branches = torch.nn.ModuleList([
            OneToMultiBranch([
                Identity(),
                torch.nn.Conv2d(C, C, kernel_size=(K, K), padding=(K - 1) // 2, stride=2),
                torch.nn.Conv2d(C, C, kernel_size=(K, K), padding=(K - 1) // 2, stride=4),
                torch.nn.Conv2d(C, C, kernel_size=(K, K), padding=(K - 1) // 2, stride=8)
            ]),
            OneToMultiBranch([
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Identity(),
                torch.nn.Conv2d(C, C, kernel_size=(K, K), padding=(K - 1) // 2, stride=2),
                torch.nn.Conv2d(C, C, kernel_size=(K, K), padding=(K - 1) // 2, stride=4)
            ]),
            OneToMultiBranch([
                torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Identity(),
                torch.nn.Conv2d(C, C, kernel_size=(K, K), padding=(K - 1) // 2, stride=2)
            ]),
            OneToMultiBranch([
                torch.nn.Upsample(scale_factor=8, mode="bilinear", align_corners=True),
                torch.nn.Upsample(scale_factor=4, mode="bilinear", align_corners=True),
                torch.nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                Identity()
            ])
        ])

    def forward(self, xs):
        branches = [branch_func(x) for x, branch_func in zip(xs, self.bcib_branches)]

        result = [
            sum(branch[0] for branch in branches),
            sum(branch[1] for branch in branches),
            sum(branch[2] for branch in branches),
            sum(branch[3] for branch in branches),
        ]

        return result


# noinspection PyAbstractClass
class GlobalAvgPool2d(torch.nn.Module):
    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        return torch.mean(x, dim=(2, 3))


# noinspection PyAbstractClass
class cSEBlock(torch.nn.Module):
    def __init__(self, C, r):
        super().__init__()

        self.C = C
        self.r = r

        self.c_attention = torch.nn.Sequential(
            GlobalAvgPool2d(),
            torch.nn.Linear(C, C // r),
            torch.nn.ReLU(),
            torch.nn.Linear(C // r, C),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.c_attention(x)[:, :, None, None] * x


# noinspection PyAbstractClass
class sSEBlock(torch.nn.Module):
    def __init__(self, C):
        super().__init__()

        self.C = C

        self.s_attention = torch.nn.Sequential(
            torch.nn.Conv2d(C, 1, kernel_size=(1, 1)),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        return self.s_attention(x) * x


# noinspection PyAbstractClass
class csSEBlock(torch.nn.Module):
    def __init__(self, C, r, reduction='max'):
        super().__init__()

        self.C = C
        self.r = r

        self.reduction = reduction

        self.s_block = sSEBlock(C=C)
        self.c_block = cSEBlock(C=C, r=r)

    def forward(self, x):
        if self.reduction == 'max':
            return torch.max(self.c_block(x), self.s_block(x))
        if self.reduction == 'sum':
            return self.c_block(x) + self.s_block(x)
        if self.reduction == 'mul':
            return self.c_block(x) * self.s_block(x)


# noinspection PyAbstractClass
class SegmentationBackbone(torch.nn.Module):
    def __init__(self, model_cls, model_args, model_kwargs):
        super().__init__()

        self.layers = model_cls(*model_args, **model_kwargs).features

    def forward_debug(self, x):
        intermediate = []
        current_channels, current_w, current_h = 0, 0, 0
        prev_channels, prev_w, prev_h = x.shape[1], x.shape[2], x.shape[3]

        print(0, x[0].shape if isinstance(x, tuple) else x.shape)
        for idx, layer in enumerate(self.layers):
            x_new = layer(x)

            if isinstance(x_new, tuple):
                current_channels = x_new[0].shape[1] + x_new[1].shape[1]
                current_w, current_h = x_new[0].shape[2], x_new[0].shape[3]
            else:
                current_channels = x_new.shape[1]
                current_w, current_h = x_new.shape[2], x_new.shape[3]

            if current_w != prev_w or current_h != prev_h:
                intermediate.append([idx, [prev_channels, prev_w, prev_h]])
                prev_channels, prev_w, prev_h = current_channels, current_w, current_h

            x = x_new
            prev_channels = current_channels
            print(idx + 1, x[0].shape if isinstance(x, tuple) else x.shape)

        intermediate.append([len(self.layers), [current_channels, current_w, current_h]])

        return intermediate[1:5]

    def forward(self, x):
        intermediate = []
        prev_w, prev_h = x.shape[2], x.shape[3]

        for idx, layer in enumerate(self.layers):
            x_new = layer(x)

            if isinstance(x_new, tuple):
                current_w, current_h = x_new[0].shape[2], x_new[0].shape[3]
            else:
                current_w, current_h = x_new.shape[2], x_new.shape[3]

            if current_w != prev_w or current_h != prev_h:
                intermediate.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x)
                prev_w, prev_h = current_w, current_h

            x = x_new

        intermediate.append(torch.cat(x, dim=1) if isinstance(x, tuple) else x)

        return intermediate[1:5]


# noinspection PyAbstractClass
class BICANet(torch.nn.Module):
    def __init__(
            self,
            ccpb_parameters,
            bcib_c, bcib_k,
            se_c, se_r, se_reduction,
            upscale_factor,
            mcfb_l, mcfb_k, mcfb_h, mcfb_w
    ):
        super().__init__()

        self.bicanet = torch.nn.Sequential(
            AdaptiveConvolution(ccpb_parameters),
            BCIB(C=bcib_c, K=bcib_k),
            MultiResolutionFusion(),
            csSEBlock(C=se_c, r=se_r, reduction=se_reduction),
            torch.nn.Upsample(scale_factor=upscale_factor, mode="bilinear", align_corners=True),
            MCFB(L=mcfb_l, K=mcfb_k, H=mcfb_h, W=mcfb_w)
        )

    def forward(self, xs):
        return self.bicanet(xs)


# noinspection PyAbstractClass
class BICANetBase(torch.nn.Module):
    _loaders = {
        'densenet121': [
            torch.hub.load, ['pytorch/vision:v0.6.0', 'densenet121'], {'pretrained': True}
        ],
        'densenet161': [
            torch.hub.load, ['pytorch/vision:v0.6.0', 'densenet161'], {'pretrained': True}
        ],
        'densenet169': [
            torch.hub.load, ['pytorch/vision:v0.6.0', 'densenet169'], {'pretrained': True}
        ],
        'densenet201': [
            torch.hub.load, ['pytorch/vision:v0.6.0', 'densenet201'], {'pretrained': True}
        ],

        'dpn68': [
            pretrainedmodels.dpn68, [], {'num_classes': 1000, 'pretrained': 'imagenet'}
        ],
        'dpn98': [
            pretrainedmodels.dpn98, [], {'num_classes': 1000, 'pretrained': 'imagenet'}
        ],
        'dpn131': [
            pretrainedmodels.dpn131, [], {'num_classes': 1000, 'pretrained': 'imagenet'}
        ]
    }

    _ccpb_parameters = {
        'densenet121': [
            (64, 30, 10, 10, 10),
            (256, 120, 40, 40, 40),
            (512, 240, 80, 80, 80),
            (1024, 480, 160, 160, 160)
        ],
        'densenet161': [
            (96, 45, 15, 15, 15),
            (384, 180, 60, 60, 60),
            (768, 360, 120, 120, 120),
            (2112, 990, 330, 330, 330)
        ],
        'densenet169': [
            (64, 30, 10, 10, 10),
            (256, 120, 40, 40, 40),
            (512, 240, 80, 80, 80),
            (1280, 600, 200, 200, 200)
        ],
        'densenet201': [
            (64, 30, 10, 10, 10),
            (256, 120, 40, 40, 40),
            (512, 240, 80, 80, 80),
            (1792, 840, 280, 280, 280)
        ],

        'dpn68': [
            (144, 60, 20, 20, 20),
            (320, 120, 40, 40, 40),
            (704, 300, 100, 100, 100),
            (832, 360, 120, 120, 120)
        ],
        'dpn98': [
            (336, 120, 40, 40, 40),
            (768, 300, 100, 100, 100),
            (1728, 840, 280, 280, 280),
            (2688, 1200, 400, 400, 400)
        ],
        'dpn131': [
            (352, 120, 40, 40, 40),
            (832, 360, 120, 120, 120),
            (1984, 840, 280, 280, 280),
            (2688, 1200, 400, 400, 400)
        ]
    }

    def __init__(self,
                 model_name,
                 num_classes,
                 l_factor, bcib_k,
                 se_r, se_reduction,
                 upscale_factor,
                 mcfb_k, mcfb_h, mcfb_w):
        super().__init__()

        L = l_factor * num_classes

        self.backbone = SegmentationBackbone(*self._loaders[model_name])
        self.bicanet = BICANet(
            ccpb_parameters=[(*_, L) for _ in self._ccpb_parameters[model_name]],
            bcib_c=L, bcib_k=bcib_k,
            se_c=4 * L, se_r=se_r, se_reduction=se_reduction,
            upscale_factor=upscale_factor,
            mcfb_l=L, mcfb_k=mcfb_k, mcfb_h=mcfb_h, mcfb_w=mcfb_w
        )

    def forward(self, x):
        xs = self.backbone(x)

        return self.bicanet(xs)

    def eval(self):
        return super().eval()

    def train(self, mode=True):
        return super().train(mode)


# noinspection PyAbstractClass
class BiCADenseNet(BICANetBase):
    def __init__(self, num_classes, arch="densenet161", l_factor=1, mcfb_h=64, mcfb_w=64):
        super().__init__(
            model_name=arch, num_classes=num_classes,
            l_factor=l_factor,
            bcib_k=3, se_r=2, se_reduction='max',
            upscale_factor=2,
            mcfb_k=5, mcfb_h=mcfb_h, mcfb_w=mcfb_w)


# noinspection PyAbstractClass
class BiCADPN(BICANetBase):
    def __init__(self, num_classes, arch="dpn98", l_factor=1, mcfb_h=64, mcfb_w=64):
        super().__init__(
            model_name=arch, num_classes=num_classes,
            l_factor=l_factor,
            bcib_k=3, se_r=2, se_reduction='max',
            upscale_factor=4,
            mcfb_k=5, mcfb_h=mcfb_h, mcfb_w=mcfb_w)
