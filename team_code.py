#!/usr/bin/env python
# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################
from torch.utils.data import DataLoader
from PIL.Image import Image
from sklearn.preprocessing import MultiLabelBinarizer
from torchvision.models import efficientnet_b3
import torch.nn.functional as F
from torchvision.transforms import ToTensor
from torchvision.transforms import ToPILImage
import joblib
from torch.utils.data import Dataset
from pytorch_wavelets import DWTForward
from torchvision import  transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from PIL import Image
import  pandas as pd
from helper_code import *


import warnings
from sklearn.exceptions import InconsistentVersionWarning

# 忽略 InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.
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

#预处理其数据
class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        # 获取图像路径
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # 获取标签（多标签）
        labels = self.img_labels.iloc[idx, 1:].values.astype('float')
        labels = torch.tensor(labels, dtype=torch.float32)

        # 应用图像变换
        if self.transform:
            image = self.transform(image)

        # 应用标签变换（如果有）
        if self.target_transform:
            labels = self.target_transform(labels)

        return image, labels
# Train your digitization model.

def img_proprecessing(img):
    width, height = img.size
    pixels = 300
    cropped_image = img.crop((0, pixels, width, height))
    image_data = cropped_image.convert('RGB')
    #image_data.show()
    # 分割图像为RGB三个通道
    r, g, b = image_data.split()
    # 保存红色通道图像，并将其转换为二值图像
    # 设定二值化的阈值
    threshold = 128
    # 使用阈值将红色通道转换为二值图像
    binary_r = r.point(lambda p: p > threshold and 255)
    # 设置图像保存时的DPI为300
    image_data = binary_r
    # 已经将图片消除网格
    transform = ToTensor()
    image_data = transform(image_data)  # 将PIL图像转换为Tensor

    # 添加一个批次维度，因为PyTorch模型通常期望这样
    image_data = image_data.unsqueeze(0)
    image_data = image_data.repeat(1, 3, 1, 1)  # 结果形状将是[1, 3, 1200, 2200]
    downsampling_layer = HWDownsampling(3, 3)
    output_data = downsampling_layer(image_data)
    output_data = downsampling_layer(output_data)
    output_data = F.interpolate(output_data, size=(300, 300), mode="area")
    to_img = ToPILImage()
    img_data = output_data.squeeze(0)
    image_xxxx = to_img(img_data)
    #image_xxxx.show()

    return img_data

def train_models(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Train the digitization model. If you are not training a digitization model, then you can remove this part of the code.

    if verbose:
        print('Training the digitization model...')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    print(torch.__version__)
    print(torch.cuda.is_available())
    print(torch.version.cuda)

    digitization_features = list()
    classification_features = list()
    classification_labels = list()
    #读取每个图片的标签
    dxs = list()
    image_path_list = []
    image_name_list = []

    if verbose:
        print('Extracting features and labels from the data...')

    for i in range(num_records):
        image_path_new = os.path.join(data_folder, records[i])
        images_name = get_image_files(image_path_new)
        for i in range(len(images_name)):
            image_name_list.append(os.path.join(data_folder, images_name[i]))
        image_path_list.append(image_path_new)
        #print(image_path_new)
        dx = load_labels(image_path_new)
        if dx:
            dxs.append(dx)
    #print('标签：',dxs)
    # 初始化 MultiLabelBinarizer
    mlb = MultiLabelBinarizer()
    # 对标签进行 fit
    mlb.fit(dxs)
    # 进行 transform
    encoded_labels = mlb.transform(dxs)
    print("Classes:", mlb.classes_)
    print("Encoded labels:\n", encoded_labels)
    labels_df = pd.DataFrame(encoded_labels, columns=mlb.classes_)
    df = pd.DataFrame({
        'image_names': image_name_list,
    })
    df = pd.concat([df, labels_df], axis=1)
    # Save the DataFrame as a CSV file
    df.to_csv('./annotations.csv', index=False)
    #去重操作
    df_unique = labels_df.drop_duplicates()
    df_unique.to_csv('./labels.csv', index=False)
    #是为了下一步保存进csv
    transform = transforms.Compose([
        # 插入自定义的裁剪操作
        transforms.Lambda(lambda img: img_proprecessing(img)),
        # 正则化Tensor图像
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image; this simple example uses the same features for the digitization and classification
        # tasks.
        features = extract_features(record)
        digitization_features.append(features)
        # Some images may not be labeled...
        labels = load_labels(record)

    # Train the models.
    if verbose:
        print('Training the models on the data...')

    # Train the digitization model. This very simple model uses the mean of these very simple features as a seed for a random number
    # generator.
    digitization_model = np.mean(features)

    # Train the classification model. If you are not training a classification model, then you can remove this part of the code.
    # Dataset loading
    dataset = CustomDataset(annotations_file='./annotations.csv', img_dir=data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=96, shuffle=True)
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    # 分成训练和验证集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    trainloader = DataLoader(train_dataset, batch_size=96, shuffle=True)
    validloader = DataLoader(test_dataset, batch_size=96, shuffle=False)

    # Define the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    local_weights_path = "./efficientnetb3.pth"
    model = efficientnet_b3(num_classes=len(dataset.img_labels.columns) -1 )
    model = model.to(device)

    # Loss function and optimizer
    criterion = F.binary_cross_entropy
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 30
    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    # 计算损失
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device).float()  # 确保标签是浮点数类型

            optimizer.zero_grad()

            outputs = model(inputs)  # 模型输出 logits
            loss = F.binary_cross_entropy_with_logits(outputs, labels)  # 适用于多标签分类的损失函数

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)

            # 使用 sigmoid 将 logits 转换为概率
            preds = torch.sigmoid(outputs)
            # 将概率与阈值 0.5 比较进行二值化
            preds = preds >= 0.5
            running_corrects += torch.sum(preds == labels).item()

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects / (len(trainloader.dataset) * labels.size(1))
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print(f'Epoch {epoch}/{num_epochs - 1}, Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')

    # Create a folder for the models if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the models.
    save_models(model_folder, digitization_model, model, mlb, mlb.classes_)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_models(model_folder, verbose):
    digitization_filename = os.path.join(model_folder, 'digitization_model.sav')
    digitization_model = joblib.load(digitization_filename)

    classification_filename = os.path.join(model_folder, 'classification_model.sav')
    classification_model = joblib.load(classification_filename)
    classification_weights_filename = os.path.join(model_folder, 'classification_model.pth')
    classification_model['model'].load_state_dict(torch.load(classification_weights_filename))


    return digitization_model, classification_model

# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you did not train one of the models, then you can return None for the model.
def run_models(record, digitization_model, classification_model, verbose):
    # Run the digitization model; if you did not train this model, then you can set signal = None.

    # Load the digitization model.
    digitization_model = digitization_model['model']

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Generate "random" waveforms using the a random seed from the features.
    seed = int(round(digitization_model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1, high=1, size=(num_samples, num_signals))

    # Run the classification model; if you did not train this model, then you can set labels = None.
    csv_data = pd.read_csv('./labels.csv', low_memory=False)  # 防止弹出警告
    csv_df = pd.DataFrame(csv_data)
    # Load the classification model and classes.

    model = classification_model['model']
    classes = classification_model['classes']
    mlb = classification_model['mlb']
    #加载模型相关
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 确保模型在设备上
    model.eval()
    model = model.to(device)
    # print(model.state_dict())
    running_loss = 0.0
    running_corrects = 0
    # 确定图像预处理
    transform = transforms.Compose([
        # 插入自定义的裁剪操作
        transforms.Lambda(lambda img: img_proprecessing(img)),
        # 正则化Tensor图像
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 找到所有记录
    # print(record)
    all_preds = []
    # dx = []
    image = load_images(record)[0].convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    #print(classes)
    with torch.no_grad():
        outputs = model(image_tensor)  # 获取模型输出
        min_loss = float('inf')
        min_loss_index = -1

        for index, row in csv_df.iterrows():
            label_tensor = torch.tensor(row.values, dtype=torch.float32).to(device)
            # 计算交叉熵损失torch.nn.MultiLabelSoftMarginLoss
            loss_fn = nn.BCEWithLogitsLoss()
            # 计算损失
            loss = loss_fn(outputs.to(device), label_tensor.unsqueeze(0).to(device))

            if loss < min_loss:
                min_loss = loss
                min_loss_index = index

        # 如果使用了 MultiLabelBinarizer
        if mlb:
            last_label_df = csv_df.iloc[min_loss_index].to_frame().T.values
            if np.array_equal(last_label_df, np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])):
                last_label_df = np.array([[0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]])
            print(last_label_df)
            selected_labels = mlb.inverse_transform(last_label_df)
            selected_labels = [','.join(labels) for labels in selected_labels]

        print('Predictions:' ,selected_labels)  # 假设每轮循环处理的是一张图片

    return signal, selected_labels
################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
def extract_features(record):
    mean = 200
    std = 50
    return np.array([mean, std])

# Save your trained models.
def save_models(model_folder, digitization_model=None, classification_model=None,mlb = None , classes=None):
    if digitization_model is not None:
        d = {'model': digitization_model}
        filename = os.path.join(model_folder, 'digitization_model.sav')
        joblib.dump(d, filename, protocol=0)

    if classification_model is not None:
        filename = os.path.join(model_folder, 'classification_model.pth')
        torch.save(classification_model.state_dict(), filename)
        d = {'model': classification_model,'mlb': mlb , 'classes': classes}
        filename = os.path.join(model_folder, 'classification_model.sav')
        joblib.dump(d, filename, protocol=0)
