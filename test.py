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
    text = load_header("E:\cinc2024\python-example-2024-main\ptb-xl/records100/05000/05000_lr")
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
    return age, sex, weight, height





train_dx_model("E:\cinc2024\python-example-2024-main\ptb-xl/records100/05000", "E:\cinc2024\python-example-2024-main\model", verbose=1) ### Teams: Implement this function!!!

#load_dx_model("E:\cinc2024\python-example-2024-main\model",verbose=1)
#run_dx_model(load_dx_model("E:\cinc2024\python-example-2024-main\model",verbose=1),"E:\cinc2024\python-example-2024-main\ptb-xl/records100/00000_test","E:\cinc2024\python-example-2024-main/test_outputs", verbose=1)