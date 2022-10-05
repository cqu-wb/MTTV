#!/usr/bin/env python3

import torch
import torch.nn as nn


class GlobalVisualFeatureEncoder(nn.Module):
    def __init__(self, args):
        super(GlobalVisualFeatureEncoder, self).__init__()
        self.args = args
        pool_func = (
            nn.AdaptiveAvgPool2d
            if args.img_embed_pool_type == "avg"
            else nn.AdaptiveMaxPool2d
        )
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)
        self.num_embeds = args.global_image_embeds
        assert self.num_embeds in [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 25, 36]

        if self.num_embeds in [1, 2, 3, 5, 7]:
            self.pool = pool_func((self.num_embeds, 1))
        elif self.num_embeds == 4:
            self.pool = pool_func((2, 2))
        elif self.num_embeds == 6:
            self.pool = pool_func((3, 2))
        elif self.num_embeds == 8:
            self.pool = pool_func((4, 2))
        elif self.num_embeds == 9:
            self.pool = pool_func((3, 3))
        elif self.num_embeds == 16:
            self.pool = pool_func((4, 4))
        elif self.num_embeds == 25:
            self.pool = pool_func((5, 5))
        elif self.num_embeds == 36:
            self.pool = pool_func((6, 6))

    def forward(self, x):
        # Bx2048x7x7 -> Bx2048xN -> BxNx2048  -> BxNx768
        x = self.pool(x)
        out = torch.flatten(x, start_dim=2)
        out = out.transpose(1, 2).contiguous()
        out = self.img_embeddings(out)
        return out


class RegionVisualFeatureEncoder(nn.Module):
    def __init__(self, args):
        super(RegionVisualFeatureEncoder, self).__init__()
        self.img_embeddings = nn.Linear(args.img_hidden_sz, args.hidden_sz)

    def forward(self, x):
        return self.img_embeddings(x)


class MixImageEncoder(nn.Module):
    def __init__(self, args):
        super(MixImageEncoder, self).__init__()
        self.fast = GlobalVisualFeatureEncoder(args)
        self.region = RegionVisualFeatureEncoder(args)

    def forward(self, global_feature, region_feature):
        global_feature = self.fast(global_feature)
        region_feature = self.region(region_feature)
        out = torch.cat([global_feature, region_feature], dim=1)
        return out

