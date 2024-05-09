from PIL import Image
import os

def split_and_save_rgb(image_data, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 从二进制数据加载图像
    img = Image.open(image_data)

    # 确保图像是RGB模式
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 分割图像为RGB三个通道
    r, g, b = img.split()

    # 保存红色通道图像，并将其转换为二值图像
    # 设定二值化的阈值
    threshold = 128
    # 使用阈值将红色通道转换为二值图像
    binary_r = r.point(lambda p: p > threshold and 255)

    # 设置图像保存时的DPI为300
    dpi_setting = (300, 300)  # DPI设置为300

    # 保存红色通道和红色通道的二值版本
    r.save(os.path.join(output_folder, "Red_Channel.jpg"), "JPEG", dpi=dpi_setting)
    binary_r.save(os.path.join(output_folder, "Red_Channel_Binary.jpg"), "JPEG", dpi=dpi_setting)

    # 返回输出文件夹中的文件列表，以验证保存成功
    return os.listdir(output_folder)

# 定义输出文件夹的路径
output_folder_path = "E:\cinc2024\python-example-2024-main"

# 执行函数
split_result = split_and_save_rgb("E:\cinc2024\python-example-2024-main\ptb-xl/records100/17000/17119_lr-0.png", output_folder_path)
