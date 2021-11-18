import torch
from torch import nn

# 参考AlexNet设计
class NiN(nn.Module):
    def __init__(self, num_labels):
        super(NiN, self).__init__()
        self.net = nn.Sequential(
            self.nin_block(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=2),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=96, out_channels=256, kernel_size=5, stride=1, padding=2),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.Dropout(p=0.5),
            nn.MaxPool2d(kernel_size=3, stride=2),
            self.nin_block(in_channels=384, out_channels=num_labels, kernel_size=3, stride=1, padding=1),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.init_weight()

    def forward(self,x):
        return self.net(x)

    def init_weight(self):
        for layer in self.net:
            if isinstance(layer, nn.Conv2d):
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(layer.bias, 0)

    def nin_block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(1, 1), stride=(1, 1)),
            nn.ReLU()
        )

    def test_output_shape(self):
        test_img = torch.rand(size=(1, 3, 227, 227), dtype=torch.float32)
        for layer in self.net:
            test_img = layer(test_img)
            print(layer.__class__.__name__, 'output shape: \t', test_img.shape)

# nin = NiN(num_labels=5)
# nin.test_output_shape()
