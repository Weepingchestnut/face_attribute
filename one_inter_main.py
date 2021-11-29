import argparse
import base64
import json
import time
import warnings
from datetime import datetime
from io import BytesIO

import PIL.Image
import cv2
import torch
import torchvision.transforms as transforms
import tornado
from torch.backends import cudnn

import torch.utils.data as data
import model as models
from utils.datasets import attr_nums
from utils.display import *

warnings.filterwarnings('ignore')
# sys.path.append('model/')

parser = argparse.ArgumentParser(description='Face Attribute Framework')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')

args = parser.parse_args()

resume_path = {
    'celeba': 'checkpoint/4_ce_mA89-72.pth.tar',
    'fairface': 'checkpoint/fairface_36_76-66.pth.tar',
    'face_mask': 'checkpoint/2_fm_psize128_mA_99-99.pth.tar'
}

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor(),
    normalize
])


def prepare_model():
    # create model
    ce_model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['celeba'])
    fair_model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['fairface'])
    mask_model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['face_mask'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    ce_model = torch.nn.DataParallel(ce_model).cuda()
    fair_model = torch.nn.DataParallel(fair_model).cuda()
    mask_model = torch.nn.DataParallel(mask_model).cuda()

    # optionally resume from a checkpoint
    ce_checkpoint = torch.load(resume_path['celeba'])
    fair_checkpoint = torch.load(resume_path['fairface'])
    mask_checkpoint = torch.load(resume_path['face_mask'])

    ce_model.load_state_dict(ce_checkpoint['state_dict'])
    fair_model.load_state_dict(fair_checkpoint['state_dict'])
    mask_model.load_state_dict(mask_checkpoint['state_dict'])

    cudnn.benchmark = False
    cudnn.deterministic = True

    return ce_model, fair_model, mask_model


celeba_model, fairface_model, face_mask_model = prepare_model()


def get_attr_list(c_output_list, f_output_list, m_output_list):
    attr_list = ['0', '未知', '未知', '未知', '未知', '0', '0', '0']

    f_gender = f_output_list[0]
    age = [max(f_output_list[1], f_output_list[2], f_output_list[3]),
           max(f_output_list[4], f_output_list[5]),
           max(f_output_list[6], f_output_list[7]),
           max(f_output_list[8], f_output_list[9])]
    race = [f_output_list[10], f_output_list[11], f_output_list[12], f_output_list[13],
            f_output_list[14], f_output_list[15], f_output_list[16]]
    hair_style = [c_output_list[1], c_output_list[2], c_output_list[3]]
    hair_color = [c_output_list[4], c_output_list[5], c_output_list[6], c_output_list[7]]
    glasses = c_output_list[8]
    hat = c_output_list[9]

    if f_gender > 0.5:
        attr_list[0] = '1'

    attr_list[1] = age_dict[age.index(max(age))]

    if max(race) < 0.5:     # 未知
        pass
    else:
        attr_list[2] = race_dict[race.index(max(race))]

    if max(hair_style) < 0.1:
        pass
    else:
        attr_list[3] = hair_style_dict[hair_style.index(max(hair_style))]

    if max(hair_color) < 0.1:
        pass
    else:
        attr_list[4] = hair_color_dict[hair_color.index(max(hair_color))]

    if glasses > 0.1:
        attr_list[5] = '1'

    if hat > 0.5:
        attr_list[6] = '1'

    if m_output_list.index(max(m_output_list)) == 0:
        attr_list[7] = '1'

    return attr_list


# one image inference
def face_attr(img_input):
    celeba_model.eval()
    fairface_model.eval()
    face_mask_model.eval()
    # 图片预处理
    if not PIL.Image.isImageType(img_input):
        img = Image.fromarray(cv2.cvtColor(img_input, cv2.COLOR_BGR2RGB))
    else:
        img = img_input
    img = transform_test(img)
    # + batch dim
    img = torch.unsqueeze(img, 0)
    # print("input.size() = {}".format(img.size()))
    img = img.cuda(non_blocking=True)
    # 模型推理
    c_output, f_output, m_output = celeba_model(img), fairface_model(img), face_mask_model(img)
    # maximum voting
    c_output = torch.max(torch.max(torch.max(c_output[0], c_output[1]), c_output[2]), c_output[3])
    f_output = torch.max(torch.max(torch.max(f_output[0], f_output[1]), f_output[2]), f_output[3])
    m_output = torch.max(torch.max(torch.max(m_output[0], m_output[1]), m_output[2]), m_output[3])
    c_output = torch.sigmoid(c_output.data).cpu().numpy()
    f_output = torch.sigmoid(f_output.data).cpu().numpy()
    m_output = torch.sigmoid(m_output.data).cpu().numpy()
    c_output_list = c_output[0].tolist()
    f_output_list = f_output[0].tolist()
    m_output_list = m_output[0].tolist()
    # 置信度微调
    if max(f_output_list[10], f_output_list[11], f_output_list[12], f_output_list[14],
           f_output_list[15], f_output_list[16]) > 0.8:
        pass
    else:
        f_output_list[13] = f_output_list[13] * 1.5
    # 返回人脸属性字典
    return get_attr_list(c_output_list, f_output_list, m_output_list)


if __name__ == '__main__':
    img = cv2.imread('test_data/shisuo_face_test/face_于洪-于洪广场家乐福-太湖街_123.305414_41.794836_20210416113449_0.99417347.jpg')
    print(face_attr(img))

