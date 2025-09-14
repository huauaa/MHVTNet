import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

def chunk(x, n):
    x_feat=torch.chunk(x.features, n, dim=1)
    x_l=[]
    for i in range(n):
        x_l.append(x.replace_feature(x_feat[i]))
    return x_l


    
class Bag(nn.Module):
    def __init__(self, channels, reduction=4):
        super(Bag, self).__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # 2D池化
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)


        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, p, i, d):
       
        d = d.sum(dim=2)
                
        avg_context = self.global_avg_pool(d) 
        max_context = self.global_max_pool(d)

        combined_context = avg_context + max_context
        edge_att = self.fc(combined_context).unsqueeze(2)

        return edge_att * p + (1 - edge_att) * i

class conv_block(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 kernel_size=(3, 3, 3),
                 stride=(1, 1, 1),
                 padding=(1, 1, 1),
                 dilation=(1, 1, 1),
                 norm_type='bn',
                 activation=True,
                 use_bias=True,
                 groups=1
                 ):
        super().__init__()
        self.conv = nn.Conv3d(in_channels=in_features,
                              out_channels=out_features,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=use_bias,
                              groups=groups)

        self.norm_type = norm_type
        self.act = activation

        if self.norm_type == 'gn':
            self.norm = nn.GroupNorm(32 if out_features >= 32 else out_features, out_features)
        if self.norm_type == 'bn':
            self.norm = nn.BatchNorm3d(out_features)
        if self.act:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        if self.norm_type is not None:
            x = self.norm(x)
        if self.act:
            x = self.relu(x)
        return x

class HCVF(nn.Module):
    def __init__(self, in_features, out_features) -> None:
         super().__init__()
         self.bag = Bag(8)
        
         self.tail_conv = nn.Sequential(
             conv_block(in_features=out_features,
                        out_features=out_features,
                        kernel_size=(1, 1, 1),
                        padding=(0, 0, 0),
                        norm_type=None,
                        activation=False)
         )
        
         self.bns = nn.BatchNorm3d(out_features)

         self.relu = nn.ReLU()
         self.fc = nn.Conv3d(out_features, in_features, kernel_size=1, bias=False)
         
         self.up=spconv.SparseConvTranspose3d(64, 32, kernel_size=(4,2,2), stride=(1,2,2), padding=0)
         self.down=spconv.SparseConv3d(32, 32, kernel_size=3, stride=2, padding=1)

    def forward(self, x_high, x, x_low):     

        x_high = self.down(x_high)
        x_high = chunk(x_high, 4)

        x_low = self.up(x_low)
        x_low = chunk(x_low, 4)  

        x = chunk(x, 4)

        x0 = self.bag(x_low[0].dense(), x_high[0].dense(), x[0].dense())
        x1 = self.bag(x_low[1].dense(), x_high[1].dense(), x[1].dense())
        x2 = self.bag(x_low[2].dense(), x_high[2].dense(), x[2].dense())
        x3 = self.bag(x_low[3].dense(), x_high[3].dense(), x[3].dense())

        x = torch.cat((x0, x1, x2, x3), dim=1)
        
        x = self.tail_conv(x)
        
        x = self.bns(x)
        x = self.fc(x)
        x = self.relu(x)

        return x

    
    
