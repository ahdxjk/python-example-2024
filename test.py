import argparse
import sys

from helper_code import *
#from run_model import *
from team_code import *
import torch
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
import PIL
import re


def extract_info():
    text = load_header("Y:\python-example-2024-main\ptb-xl/records500/00000/00001_hr")
    # 使用正则表达式查找各项数据，返回的是一个元组
    age_match = re.search(r"#Age: (\d+)", text)
    sex_match = re.search(r"#Sex: (\w+)", text)
    weight_match = re.search(r"#Weight: (\d+)", text)
    height_match = re.search(r"#Height: ([\w ]+)", text)
    # 提取年龄
    age = int(age_match.group(1)) if age_match else None
    # 提取性别，假设Sex字段后只有Male或Female，将其转换为0（男性）或1（女性）
    sex = 0 if sex_match and sex_match.group(1).lower() == 'male' else 1 if sex_match else None
    # 提取体重
    weight = int(weight_match.group(1)) if weight_match else None
    # 提取身高，如果身高为"Unknown"或者没有找到身高信息，则返回None
    height = int(height_match.group(1)) if height_match and height_match.group(1).isdigit() else None

    age = age if age is not None else -1
    sex = sex if sex is not None else -1
    weight = weight if weight is not None else -1
    height = height if height is not None else -1
    # 创建一个包含所有特征的张量
    features = torch.tensor([age, sex, weight, height], dtype=torch.int)
    return features

#train_models("Y:\python-example-2024-main\ptb-xl/records500/00000", "Y:\python-example-2024-main\model", verbose=1) ###
#print(extract_info())
info = "Y:\python-example-2024-main\ptb-xl/records500/00000"
records = find_records(info)
num_records = len(records)
for i in range(num_records):
    width = len(str(num_records))
    data_record = os.path.join(info, records[i])
    digitization_model, classification_model = load_models("Y:\python-example-2024-main\model",verbose=1)
    run_models(data_record,digitization_model, classification_model, verbose=1)