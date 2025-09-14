import torch
import torch.nn as nn


import torch
import torch.nn as nn


class DynamicWeightModule(nn.Module):
    def __init__(self, channels):
        super(DynamicWeightModule, self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(channels, channels // 4),
            nn.ReLU(inplace=True),
            nn.Linear(channels // 4, 3), 
            nn.Softmax(dim=1) 
        )

    def forward(self, x):
        weights = self.fc(x) 
        return weights[:, 0], weights[:, 1], weights[:, 2]

class CoordAtt3D(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt3D, self).__init__()
        self.apool_d = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.apool_h = nn.AdaptiveAvgPool3d((1, None, 1))
        self.apool_w = nn.AdaptiveAvgPool3d((1, 1, None))
        
        self.mpool_d = nn.AdaptiveMaxPool3d((None, 1, 1))
        self.mpool_h = nn.AdaptiveMaxPool3d((1, None, 1))
        self.mpool_w = nn.AdaptiveMaxPool3d((1, 1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Sequential(
            nn.Conv3d(inp, mip, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm3d(mip),
            nn.ReLU()
        )
        

        self.dynamic_weight = DynamicWeightModule(inp)
        
        self.conv_d = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_h = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv3d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):

        identity = x
        B, C, D, H, W = x.size()

        w_d, w_h, w_w = self.dynamic_weight(x)
        w_d = w_d.view(B, 1, 1, 1, 1)
        w_h = w_h.view(B, 1, 1, 1, 1)
        w_w = w_w.view(B, 1, 1, 1, 1)

        xa_d = self.apool_d(x)
        xa_h = self.apool_h(x).permute(0, 1, 3, 2, 4)
        xa_w = self.apool_w(x).permute(0, 1, 4, 2, 3)
        
        xm_d = self.mpool_d(x)
        xm_h = self.mpool_h(x).permute(0, 1, 3, 2, 4)
        xm_w = self.mpool_w(x).permute(0, 1, 4, 2, 3)
        
        x_d = xm_d + xa_d
        x_h = xm_h + xa_h
        x_w = xm_w + xa_w
        
        y = torch.cat([x_d, x_h, x_w], dim=2)
        y = self.conv1(y)

        x_d, x_h, x_w = torch.split(y, [D, H, W], dim=2)
        x_h = x_h.permute(0, 1, 3, 2, 4)
        x_w = x_w.permute(0, 1, 4, 3, 2)
        
        a_d = self.conv_d(x_d).sigmoid() * w_d
        a_h = self.conv_h(x_h).sigmoid() * w_h
        a_w = self.conv_w(x_w).sigmoid() * w_w

        out = identity * a_d * a_h * a_w
        return out