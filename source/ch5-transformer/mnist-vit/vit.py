from torch import nn
import torch
class ViT(nn.Module):
    def __init__(self, img_size=28, emb_size=16, patch_size=4, in_chans=1, depth=3, heads=2, classify_features=None):
        super().__init__()
        self.patch_size = patch_size
        self.patch_count = (img_size // self.patch_size) ** 2
        self.conv = nn.Conv2d(in_channels=in_chans, out_channels=patch_size ** 2, kernel_size=patch_size, padding=0,
                              stride=patch_size)
        self.patch_emb = nn.Linear(in_features=patch_size ** 2, out_features=emb_size)
        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_emb = nn.Parameter(torch.rand(1, self.patch_count + 1, emb_size))
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=heads, batch_first=True)
        self.tranformer_enc = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.do_classify = False
        if classify_features is not None:
            self.do_classify = True
            self.cls_linear = nn.Linear(in_features=emb_size, out_features=classify_features)

    def forward(self, x):  # (batch_size,channel=1,width=28,height=28)
        x1 = self.conv(x)  # (batch_size,channel=16,width=7,height=7)
        x = x1.view(x1.size(0), x1.size(1), self.patch_count)  # (batch_size,channel=16,seq_len=49)
        x = x.permute(0, 2, 1)  # (batch_size,seq_len=49,channel=16)
        x = self.patch_emb(x)  # (batch_size,seq_len=49,emb_size)
        cls_token = self.cls_token.expand(x.size(0), 1, x.size(2))  # (batch_size,1,emb_size)
        x = torch.cat((cls_token, x), dim=1)  # add [cls] token
        x = self.pos_emb + x
        y = self.tranformer_enc(x)
        if self.do_classify:
            t = y[:, 0, :]
            r = self.cls_linear(t)
            return r
        else:
            return y


if __name__ == '__main__':
    vit = ViT(classify_features=10)
    x = torch.rand(5, 1, 28, 28)
    from hiq.vis import print_model
    print_model(vit)
    y = vit(x)
    assert y.shape == torch.Size([5, 10])
