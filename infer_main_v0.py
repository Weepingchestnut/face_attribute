"""
    包含口罩检测的推理模型
"""
import argparse
import base64
import json
from datetime import datetime
import os
import sys
import time
import warnings
from io import BytesIO
from pprint import pprint

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
import tornado
from PIL import Image
from torch.backends import cudnn

import model as models
# from mask_infer import inference
from utils.anchor_decode import decode_bbox
from utils.anchor_generator import generate_anchors
from utils.datasets import attr_nums, get_test_data
from utils.display import *
from utils.nms import single_class_non_max_suppression

# from utils.pytorch_loader import load_pytorch_model, pytorch_inference

warnings.filterwarnings('ignore')
sys.path.append('model/')

parser = argparse.ArgumentParser(description='Face Attribute Framework')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--test_data_path', default='test_data/shisuo_face_test', type=str, required=False, help='(default=%(default)s)')

args = parser.parse_args()

test_data_path = '/data2/face_data/RAP/RAP_dataset'
resume_path = {
    'celeba': 'your_pathceleba/inception_iccv/3_ce_psize128_mA_89-19.pth.tar',
    'fairface': 'your_pathfairface/inception_iccv/11_fair_psize128_76-62.pth.tar',
    'face_mask': 'your_pathface_mask/inception_iccv/2.pth.tar',
    'mask': 'model/model360.pth'
}
test_path = 'test_data/shisuo/face_和平-美国领事馆_123.427057_41.783389_20210415171509_0.9848361.jpg'
save_path = 'test_data/save'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
transform_test = transforms.Compose([
    transforms.Resize(size=(128, 128)),
    transforms.ToTensor(),
    normalize
])

# feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
feature_map_sizes = [[45, 45], [23, 23], [12, 12], [6, 6], [4, 4]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)


def prepare_model():
    # create model
    ce_model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['celeba'])
    fair_model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['fairface'])
    mask_model = models.__dict__['inception_iccv'](pretrained=True, num_classes=attr_nums['face_mask'])

    # for training on multiple GPUs.
    # Use CUDA_VISIBLE_DEVICES=0,1 to specify which GPUs to use
    ce_model = torch.nn.DataParallel(ce_model).cuda()
    fair_model = torch.nn.DataParallel(fair_model).cuda()
    face_mask_model = torch.nn.DataParallel(mask_model).cuda()

    # optionally resume from a checkpoint
    ce_checkpoint = torch.load(resume_path['celeba'])
    fair_checkpoint = torch.load(resume_path['fairface'])
    face_mask_checkpoint = torch.load(resume_path['face_mask'])
    # mask_model = load_pytorch_model(resume_path['mask'])
    mask_model = torch.load(resume_path['mask'])
    mask_model.to(device)

    ce_model.load_state_dict(ce_checkpoint['state_dict'])
    fair_model.load_state_dict(fair_checkpoint['state_dict'])
    face_mask_model.load_state_dict(face_mask_checkpoint['state_dict'])
    # if os.path.isfile(resume_path):
    #     checkpoint = torch.load(resume_path)
    #     ce_model.load_state_dict(checkpoint['state_dict'])
    # else:
    #     print("=> no checkpoint found at '{}'".format(resume_path))
    cudnn.benchmark = False
    cudnn.deterministic = True
    # return ce_model, fair_model
    return ce_model, fair_model, face_mask_model, mask_model


# celeba_model, fairface_model, facemask_model = prepare_model()
celeba_model, fairface_model, face_mask_model, facemask_model = prepare_model()


# class base64_api(tornado.web.RequestHandler):
#     def initialize(self, gconf):
#         self.config = gconf
#         self.pool = gconf.get("threadpool", None)
#
#     @tornado.web.asynchronous
#     @tornado.gen.coroutine
#     def post(self, *aegs, **kwargs):
#         request = json.loads(self.request.body)
#         img_id = request.get('img_id', '')
#         base64_code = request.get('base64_code', '')
#
#         start_time = time.time()
#         try:
#             image = base64_to_pil(base64_code)
#             attr_dict = face_attr(imag, celeba_model, fairface_model, facemask_model)
#             stat = True
#         except:
#             stat = False
#         end_time = time.time()
#
#         response = dict()
#         # response["pedestrian_attribute"] = dict()
#         if not stat:
#             response["message"] = '提取失败'
#         else:
#             response["message"] = '提取成功'
#             response["img_id"] = img_id
#             response["pedestrian_attribute"] = attr_dict
#         response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
#         print(response)
#         self.write(response)


def celeba_face_attr(img_input, model):
    model.eval()
    input = transform_test(img_input)
    input = torch.unsqueeze(input, 0)
    print("input.size() = {}".format(input.size()))
    input = input.cuda(non_blocking=True)
    # print("output = model(input)")
    output = model(input)
    # maximum voting
    if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
    output = torch.sigmoid(output.data).cpu().numpy()
    # print("output = {}".format(output))
    output_list = output[0].tolist()
    print("output_list = {}".format(output_list))
    return celeba_attr_dict(output_list)


def fair_face_attr(img_input, model):
    model.eval()
    input = transform_test(img_input)
    input = torch.unsqueeze(input, 0)
    print("input.size() = {}".format(input.size()))
    input = input.cuda(non_blocking=True)
    # print("output = model(input)")
    output = model(input)
    # maximum voting
    if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
    output = torch.sigmoid(output.data).cpu().numpy()
    # print("output = {}".format(output))
    output_list = output[0].tolist()
    print("output_list = {}".format(output_list))
    return fairface_attr_dict(output_list)


def mask_face_attr_iccv(img_input, model):
    model.eval()
    input = transform_test(img_input)
    input = torch.unsqueeze(input, 0)
    print("input.size() = {}".format(input.size()))
    input = input.cuda(non_blocking=True)
    # print("output = model(input)")
    output = model(input)
    # maximum voting
    if type(output) == type(()) or type(output) == type([]):
        output = torch.max(torch.max(torch.max(output[0], output[1]), output[2]), output[3])
    output = torch.sigmoid(output.data).cpu().numpy()
    # print("output = {}".format(output))
    output_list = output[0].tolist()
    print("output_list = {}".format(output_list))
    return mask_face_attr_dict(output_list)


def face_attr(img_input, c_model, f_model, m_model):
    c_model.eval()
    f_model.eval()
    m_model.eval()
    img = transform_test(img_input)
    # + batch dim
    img = torch.unsqueeze(img, 0)
    # print("input.size() = {}".format(img.size()))
    img = img.cuda(non_blocking=True)
    c_output, f_output, m_output = c_model(img), f_model(img), m_model(img)
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
    # print("c_output_list = {}".format(c_output_list))
    # print("f_output_list = {}".format(f_output_list))
    # class_mask = mask_face_attr(img_input, m_model)

    return face_attr_dict(c_output_list, f_output_list, m_output_list)


def mask_face_attr(img_input, model, target_shape=(360, 360)):
    image = np.array(img_input)
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 归一化到0~1
    image_exp = np.expand_dims(image_np, axis=0)  # [1, H, W, C]
    image_transposed = image_exp.transpose((0, 3, 1, 2))  # [1, C, H, W]

    input_tensor = torch.tensor(image_transposed).float().to(device)
    y_bboxes, y_scores, = model.forward(input_tensor)
    y_bboxes_output, y_cls_output = y_bboxes.detach().cpu().numpy(), y_scores.detach().cpu().numpy()
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes, bbox_max_scores, conf_thresh=0.5, iou_thresh=0.4)
    # in for is "normal operation"
    for idx in keep_idxs:
        # conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        print("idx = {}, class_id = {}".format(idx, class_id))
        return class_id
    # class_id = bbox_max_score_classes[keep_idxs[0]]
    # print("class_id = {}".format(class_id))
    # if keep_idxs = None, to prevent bug, return 1 (means nomask)
    return 1


def pil_to_base64(p264_img):
    img_buffer = BytesIO()
    p264_img.save(img_buffer, format='JPEG')
    byte_data = img_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    return base64_str


def base64_to_pil(base64_str):
    img = base64.b64decode(base64_str)
    img = BytesIO(img)
    img = Image.open(img)  # .convert('RGB')
    return img


def loop_test():
    data_path = 'test_data/aligned_output'
    a = datetime.now()
    img_num = 0
    for file in os.listdir(data_path):
        img_num += 1
        print(file)
        img_path = os.path.join(data_path, file)
        imag = Image.open(img_path).convert('RGB')
        attr_dict = face_attr(imag, celeba_model, fairface_model, facemask_model)
        pprint(attr_dict)
        print("~" * 100)
    b = datetime.now()
    during = (b - a).seconds
    print(during)
    print(img_num)
    print(img_num / during)


def batch_test(test_data_path):
    # make img_name list
    img_names = []
    for file in os.listdir(test_data_path):
        img_names.append(file)

    test_dataset = get_test_data(root=test_data_path, label=img_names, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    a = datetime.now()
    test(test_loader, celeba_model, fairface_model, face_mask_model)
    b = datetime.now()
    during = (b-a).seconds
    print("batch_size = {}".format(args.batch_size))
    print("num_workers = {}".format(args.num_workers))
    print("time = {}".format(during))
    print("infer speed = {}".format(test_dataset.__len__() / during))


def test(test_loader, c_model, f_model, m_model):
    c_model.eval()
    f_model.eval()
    m_model.eval()

    for i, _ in enumerate(test_loader):
        # print(i+1)
        input, img_name = _
        input = input.cuda(non_blocking=True)
        c_output, f_output, m_output = c_model(input), f_model(input), m_model(input)
        bs = input.size(0)
        # print("bs = {}".format(bs))

        # maximum voting
        c_output = torch.max(torch.max(torch.max(c_output[0], c_output[1]), c_output[2]), c_output[3])
        f_output = torch.max(torch.max(torch.max(f_output[0], f_output[1]), f_output[2]), f_output[3])
        m_output = torch.max(torch.max(torch.max(m_output[0], m_output[1]), m_output[2]), m_output[3])

        c_output = torch.sigmoid(c_output.data).cpu().numpy()
        f_output = torch.sigmoid(f_output.data).cpu().numpy()
        m_output = torch.sigmoid(m_output.data).cpu().numpy()

        for one_bs in range(bs):
            print("img_name: {}".format(img_name[one_bs]))

            c_output_list = c_output[one_bs].tolist()
            f_output_list = f_output[one_bs].tolist()
            m_output_list = m_output[one_bs].tolist()

            # 置信度微调
            if max(f_output_list[10], f_output_list[11], f_output_list[12], f_output_list[14],
                   f_output_list[15], f_output_list[16]) > 0.8:
                pass
            else:
                f_output_list[13] = f_output_list[13] * 1.5

            pprint(face_attr_dict(c_output_list, f_output_list, m_output_list))


# if __name__ == '__main__':
#     # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     imag = Image.open(test_path).convert('RGB')
#     # base64_code = pil_to_base64(imag)
#     # print("base64 = {}".format(base64_code))
#     # # image = base64_to_pil(image)
#     # #
#     # start_time = time.time()
#     # try:
#     #     image = base64_to_pil(base64_code)
#     #     attr_dict = face_attr(imag, celeba_model, fairface_model, facemask_model)
#     #     stat = True
#     # except:
#     #     stat = False
#     # end_time = time.time()
#     #
#     # response = dict()
#     # # response["pedestrian_attribute"] = dict()
#     # if not stat:
#     #     response["message"] = '提取失败'
#     # else:
#     #     response["message"] = '提取成功'
#     #     response["pedestrian_attribute"] = attr_dict
#     # response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
#     # print(response)
#     start_time = time.time()
#     # pprint(face_attr(imag, celeba_model, fairface_model, facemask_model))
#     pprint(face_attr(imag, celeba_model, fairface_model, face_mask_model))
#     # mask infer test
#     # print(mask_face_attr_iccv(imag, face_mask_model))
#     # # print(id2class[mask_face_attr(imag, facemask_model)])      # 0 戴口罩; 1 不戴口罩
#     end_time = time.time()
#     print(str(round((end_time - start_time), 4) * 1000) + " ms")
#     # print(predict_face_mask(imag, mask_model, device))
#     # loop_test()

if __name__ == '__main__':
    test_path = 'test_data/shisuo_face_test'
    batch_test(test_path)
