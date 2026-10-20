from torch import nn
from FT_transformer import FTTransformer


class FTC(nn.Module):
    def __init__(self,
                 input_dim,
                 num_classes,
                 attn_drpout=0,
                 ff_dropout=0.1,
                 dim=8,
                 depth=6,
                 heads=4,
                 dim_head=4,
                 ):
        super(FTC, self).__init__()

        self.ft = FTTransformer(
            input_dim=input_dim,
            dim=dim,
            dim_out=num_classes,
            depth=depth,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_drpout,
            ff_dropout=ff_dropout,
        )

        self.sigmoid = nn.Sigmoid()
        self.dropout = ff_dropout

    def forward(self, x):
        x = self.ft(x)
        x = self.sigmoid(x)
        return x
