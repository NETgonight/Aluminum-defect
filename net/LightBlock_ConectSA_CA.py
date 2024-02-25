import torch.nn as nn
import torch


def encoder_block(in_channels, out_channels, groups=1, slope=0.2):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1), groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(slope)
    )
    return block


def LightBlockEncoder(in_channels, out_channels, groups=1, slope=0.2):
    block = nn.Sequential(

        # 深度可分离卷积 进行空间特征提取，将图片下采样
        nn.Conv2d(in_channels, in_channels * groups, kernel_size=3, stride=2, padding=1, groups=groups),

        # 点卷积 进行通道维度提取，翻倍通道数
        nn.Conv2d(in_channels * groups, out_channels, kernel_size=1, stride=1, padding=0),

        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(slope)
    )
    return block


class AttentionModule(nn.Module):
    def __init__(self, channels):
        super(AttentionModule, self).__init__()
        # 通道注意力
        self.channel_attention = ChannelAttention(channels)
        # 空间注意力
        self.spatial_attention = SpatialAttention(channels)
        self.upconv = nn.ConvTranspose2d(channels*2, channels, kernel_size=1, padding=0, output_padding=0)

    def forward(self, x):
        # 通道注意力和空间注意力的拼接
        # x = self.channel_attention(x)
        x = torch.cat([self.channel_attention(x), self.spatial_attention(x)], dim=1)
        x = self.upconv(x)
        return x

# 通道注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, channels):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // 8, kernel_size=1, stride=1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, channels, kernel_size=1, stride=1)
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = torch.sigmoid(avg_out + max_out)
        return out * x


# 空间注意力模块
class SpatialAttention(nn.Module):
    def __init__(self, channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)



def decoder_block(in_channels, out_channels, groups=1, slope=0.2):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1), groups=groups),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(slope)
    )
    return block


class LightBlock_ConectSA_CA(nn.Module):
    def __init__(self, group=1):
        super(LightBlock_ConectSA_CA, self).__init__()

        # [b,3,640,480] ==> [b,6,320,240]
        self.encoder1 = LightBlockEncoder(3, 6, groups=group)

        # [b,6,320,240] ==> [b,12,160,120]
        self.encoder2 = LightBlockEncoder(6, 12, groups=group)
        self.skip2 = AttentionModule(12)

        # [b,12,160,120] ==> [b,24,80,60]
        self.encoder3 = LightBlockEncoder(12, 24, groups=group)
        self.skip3 = AttentionModule(24)

        # [b,24,80,60] ==> [b,48,40,30]
        self.encoder4 = LightBlockEncoder(24, 48, groups=group)
        self.skip4 = AttentionModule(48)

        # [b,48,40,30] ==> [b,96,20,15]
        # self.bottleneck = encoder_block(48, 96, groups=group)
        # 采用深度可分离卷积替代原先减少计算量
        self.bottleneck = LightBlockEncoder(48, 96, groups=group)

        # [b,96,20,15] ==> [b,48,40,30]
        # self.decoder1 = decoder_block(96, 48, groups=group)
        # 采用深度可分离卷积替代原先减少计算量
        self.decoder1 = decoder_block(96, 48, groups=group)

        # [b,48,40,30]+[b,48,40,30] ==> [b,96,40,30] ==> [b,24,80,60]
        self.decoder2 = decoder_block(96, 24, groups=group)

        # [b,24,80,60]+[b,24,80,60] ==> [b,48,80,60] ==> [b,12,160,120]
        self.decoder3 = decoder_block(48, 12, groups=group)

        # [b,12,160,120]+[b,12,160,120] ==> [b,24,160,120] ==> [b,6,320,240]
        self.decoder4 = decoder_block(24, 6, groups=group)

        # [b,6,320,240]+[b,6,320,240] ==> [b,12,320,240] ==> [b,1,640,480]
        self.output = nn.Sequential(
            nn.ConvTranspose2d(12, 5, (3, 3), stride=(2, 2), padding=(1, 1), output_padding=(1, 1)),
            nn.BatchNorm2d(5),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        encoder1 = self.encoder1(x)  # [b,3,640,480] ==> [b,6,320,240]
        encoder2 = self.encoder2(encoder1)  # [b,6,320,240] ==> [b,12,160,120]
        encoder3 = self.encoder3(encoder2)  # [b,12,160,120] ==> [b,24,80,60]
        encoder4 = self.encoder4(encoder3)  # [b,24,80,60] ==> [b,48,40,30]

        bottleneck = self.bottleneck(encoder4)  # [b,48,40,30] ==> [b,96,20,15]

        decoder1 = self.decoder1(bottleneck)  # [b,96,20,15] ==> [b,48,40,30]

        encoder4 = self.skip4(encoder4)  # [b,48,40,30] ==> [b,96,40,30]
        decoder2 = self.decoder2(
            torch.cat([encoder4, decoder1], 1))  # [b,48,40,30]+[b,48,40,30] ==> [b,96,40,30] ==> [b,24,80,60]

        encoder3 = self.skip3(encoder3)
        decoder3 = self.decoder3(
            torch.cat([encoder3, decoder2], 1))  # [b,24,80,60]+[b,24,80,60] ==> [b,48,80,60] ==> [b,12,160,120]

        encoder2 = self.skip2(encoder2)
        decoder4 = self.decoder4(
            torch.cat([encoder2, decoder3], 1))  # [b,12,160,120]+[b,12,160,120] ==> [b,24,160,120] ==> [b,6,320,240]

        output = self.output(
            torch.cat([encoder1, decoder4], 1))  # [b,6,320,240]+[b,6,320,240] ==> [b,12,320,240] ==> [b,1,640,480]
        return output


if __name__ == "__main__":
    from torchinfo import summary

    model = LightBlock_ConectSA_CA()
    print(model)

    # 打印网络结构图
    summary(model, input_size=(1, 3, 640, 480), device="cpu",
            col_names=["input_size", "output_size", "num_params", 'params_percent', 'kernel_size', 'mult_adds', 'trainable'])

    # 计算参数
    from thop import profile

    input = torch.randn(1, 3, 640, 480)
    flops, parms = profile(model, inputs=(input,))
    print(f"FLOPs:{flops / 1e9}G,params:{parms / 1e6}M")

    # img = torch.randn(1, 3, 640, 480)
    # out = model(img)
    # print(out)

    # predict = out > 0.5
    # pre = predict.type(torch.int8).numpy()
    # print(pre)
