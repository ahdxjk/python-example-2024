from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from pytorch_wavelets import DWTForward
import torch.nn.functional as F
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision.transforms import ToPILImage
class HWDownsampling(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(HWDownsampling, self).__init__()
        self.wt = DWTForward(J=1, wave='haar', mode='zero')
        self.conv_bn_relu = nn.Sequential(
            nn.Conv2d(in_channel * 4, out_channel, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.wt(x)
        y_HL = yH[0][:, :, 0, ::]
        y_LH = yH[0][:, :, 1, ::]
        y_HH = yH[0][:, :, 2, ::]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        x = self.conv_bn_relu(x)
        return x


def image_look(outputdata):
    unloader = transforms.ToPILImage()
    image = output_data.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image.show()


# 打开图像
image = Image.open("E:\cinc2024\python-example-2024-main\ptb-xl/records100/17000/17119_lr-0.png")
# 设置裁剪的上边界和下边界
top = 500
bottom = 1700

# 执行裁剪操作
# left, upper, right, lower
cropped_image = image.crop((0, top, 2200, bottom))
image_data = cropped_image.convert('RGB')
# 分割图像为RGB三个通道
r, g, b = image_data.split()
# 保存红色通道图像，并将其转换为二值图像
# 设定二值化的阈值
threshold = 128
# 使用阈值将红色通道转换为二值图像
binary_r = r.point(lambda p: p > threshold and 255)
# 设置图像保存时的DPI为300
image_data = binary_r
#已经将图片消除网格
transform = ToTensor()
image_data = transform(image_data)  # 将PIL图像转换为Tensor

# 添加一个批次维度，因为PyTorch模型通常期望这样
image_data = image_data.unsqueeze(0)
image_data = image_data.repeat(1, 3, 1, 1)  # 结果形状将是[1, 3, 1200, 2200]

target_height = 224
target_width = 224

downsampling_layer = HWDownsampling(3, 3)
output_data = downsampling_layer(image_data)
output_data = downsampling_layer(output_data)
output_data = F.interpolate(output_data, size=(224, 224),mode="area")
to_img = ToPILImage()
img_data = output_data
print(img_data.shape)




