#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
from torch.utils.data import DataLoader, Dataset
import PIL
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader
from torch import nn, optim
import torch
from PIL import Image
import  pandas as pd
from helper_code import *
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt


################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments of the functions.
#
################################################################################

# Train your digitization model.
def train_digitization_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the digitization model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    # Extract the features and labels.
    if verbose:
        print('Extracting features and labels from the data...')

    features = list()

    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])

        # Extract the features from the image...
        current_features = extract_features(record)
        features.append(current_features)

    # Train the model.
    if verbose:
        print('Training the model on the data...')

    # This overly simple model uses the mean of these overly simple features as a seed for a random number generator.
    model = np.mean(features)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_digitization_model(model_folder, model)

    if verbose:
        print('Done.')
        print()


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.img_labels.iloc[idx, 0]
        image = Image.open(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label



# Train your dx classification model.


def train_dx_model(data_folder, model_folder, verbose):
    # Find data files.
    if verbose:
        print('Training the dx classification model...')
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    #print(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data was provided.')

    features = list()
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
        dx = load_dx(image_path_new)
        if dx:
            dxs.append(dx)

    if verbose and (len(image_name_list) == 0 or len(dxs) == 0):
        print("Warning: The lists image_name_list or dxs is empty. Check your data loading logic.")

    le = LabelEncoder()
    dxs = le.fit_transform(np.array(dxs).ravel())
    #print(image_name_list)
    #print(dxs)
    #####################对齐两个数组，以防有缺失值
    min_length = min(len(dxs), len(image_name_list))
    # Truncate lists to the same length
    dxs = dxs[:min_length]
    image_name_list = image_name_list[:min_length]
    #####################对齐两个数组，以防有缺失值

    df = pd.DataFrame({
        'image_names': image_name_list,
        'dxs': dxs
    })

    # Save the DataFrame as a CSV file
    df.to_csv('./annotations.csv', index=False)

    ################################################################################
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.RandomRotation(20),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(224),
        transforms.Lambda(lambda img: img.convert('RGB')),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []
    # Dataset loading
    dataset = CustomDataset(annotations_file='./annotations.csv', img_dir=data_folder, transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # 分成训练和验证集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    trainloader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    validloader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Define the model
    local_weights_path = './resnet50.pth'
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(local_weights_path))
    num_ftrs = model.fc.in_features
    num_classes = 2

    model.fc = nn.Linear(num_ftrs, num_classes)
    # Move the model to GPU if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    # 使用Adam优化器并应用L2正则化
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
    loss = torch.nn.CrossEntropyLoss()

    num_epochs = 20

    for epoch in range(num_epochs):
        # 在训练阶段
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = torch.tensor(labels).to(device)

            optimizer.zero_grad()  # Zero the gradients

            outputs = model(inputs)
            #print(outputs)
            _, preds = torch.max(outputs.data, 1)
            loss = criterion(outputs, labels)

            loss.backward()  # Backpropagation
            optimizer.step()  # 对模型进行优化

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(trainloader.dataset)
        epoch_acc = running_corrects.double() / len(trainloader.dataset)
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        print("迭代次数：",epoch)
        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # 在验证阶段
        model.eval()  # 切换模型为评估模式
        running_loss = 0.0
        running_corrects = 0

        with torch.no_grad():
            for inputs, labels in validloader:
                inputs = inputs.to(device)
                labels = torch.tensor(labels).to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs.data, 1)
                print("验证阶段的结果", preds)
                loss = criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)

        epoch_loss = running_loss / len(validloader.dataset)
        epoch_acc = running_corrects.double() / len(validloader.dataset)
        valid_losses.append(epoch_loss)
        valid_accuracies.append(epoch_acc)
        print("验证集",end=" ")
        print(f'Valid Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')



    # Save the model
    os.makedirs(model_folder, exist_ok=True)

    if verbose:
        print('Finished Training the dx classification model')

    save_dx_model(model_folder, model)






# Load your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function. If you do not train a digitization model, then you can return None.
def load_digitization_model(model_folder, verbose):
    filename = os.path.join(model_folder, 'digitization_model.sav')
    return joblib.load(filename)

# Load your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function. If you do not train a dx classification model, then you can return None.



# Run your trained digitization model. This function is *required*. You should edit this function to add your code, but do *not*
# change the arguments of this function.
def run_digitization_model(digitization_model, record, verbose):
    model = digitization_model['model']

    # Extract features.
    features = extract_features(record)

    # Load the dimensions of the signal.
    header_file = get_header_file(record)
    header = load_text(header_file)

    num_samples = get_num_samples(header)
    num_signals = get_num_signals(header)

    # For a overly simply minimal working example, generate "random" waveforms.
    seed = int(round(model + np.mean(features)))
    signal = np.random.default_rng(seed=seed).uniform(low=-1000, high=1000, size=(num_samples, num_signals))
    signal = np.asarray(signal, dtype=np.int16)

    return signal

# Run your trained dx classification model. This function is *required*. You should edit this function to add your code, but do
# *not* change the arguments of this function.


def save_dx_model(model_folder, model):
    filename = os.path.join(model_folder, 'dx_model.pth')
    torch.save(model, filename)  # Save the entire model_state


def load_dx_model(model_folder, verbose):
    # Assume the model file is named 'dx_model.pth'
    model_file = os.path.join(model_folder, 'dx_model.pth')

    # Initialize the model structure
    local_weights_path = './resnet50.pth'
    model = models.resnet50(pretrained=False)
    model.load_state_dict(torch.load(local_weights_path))
    num_ftrs = model.fc.in_features
    num_classes = 2
    model.fc = nn.Linear(num_ftrs, num_classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load the model weights
    torch.load(model_file, map_location=device)

    return model


def run_dx_model(dx_model, record, signal, verbose):
    # 确保模型在正确的设备上
    local_weights_path = './resnet50.pth'
    model = dx_model
    print(model.parameters())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # 确保模型在设备上
    model = model.to(device)
    #print(model.state_dict())
    model.eval()

    # 确定图像预处理
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    # 找到所有记录
    #print(record)
    all_preds = []
    #dx = []
    image_path = os.path.join(record + '-0.png')
    #print(image_path)
    #dx_name = os.path.join(record, record_name)
    #dx.append(load_dx(dx_name))
    #print(dx)
    image = Image.open(image_path).convert('RGB')
    #print("image",image)
    image_tensor = transform(image).unsqueeze(0).to(device)

    # 创建一个与类别索引对应的标签列表
    labels = ["normal", "abnormal"]

    with torch.no_grad():
        outputs = model(image_tensor)
        #print(outputs)
        _, preds = torch.max(abs(outputs), 1)
        # 使用索引从标签集中取出对应的标签
        preds_label = [labels[p] for p in preds]
        print(f'Image: {image_path}, Prediction: {preds_label[0]}')  # 假设每轮循环处理的是一张图片

    return preds_label



################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract features.
class ImageDataset(Dataset):
    def __init__(self, numpy_images, transform=None):
        self.images = numpy_images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = PIL.Image.fromarray(image)

        if image.mode == 'RGBA':  # Check if the image is RGBA and convert to RGB
            image = image.convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image



# Save your trained digitization model.
def save_digitization_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'digitization_model.sav')
    joblib.dump(d, filename, protocol=0)


def extract_features(record):
    images = load_image(record)
    mean = 0.0
    std = 0.0
    for image in images:
        image = np.asarray(image)
        mean += np.mean(image)
        std += np.std(image)
    return np.array([mean, std])