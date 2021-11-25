import argparse
import base64
import json
import time
import warnings
from datetime import datetime
from io import BytesIO

import torch
import torchvision.transforms as transforms
import tornado
from torch.backends import cudnn

import model as models
from utils.datasets import attr_nums
from utils.display import *

warnings.filterwarnings('ignore')
# sys.path.append('model/')

parser = argparse.ArgumentParser(description='Face Attribute Framework')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=0, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--test_data_path', default='test_data/shisuo_face_test', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('-s', '--show', dest='show', action='store_true', required=False,
                    help='show attribute in imag')

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


def isBase64(s):
    """Check s is Base64.b64encode"""
    if not isinstance(s, str) or not s:
        return False

    _base64_code = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I',
                    'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R',
                    'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a',
                    'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
                    'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's',
                    't', 'u', 'v', 'w', 'x', 'y', 'z', '0', '1',
                    '2', '3', '4', '5', '6', '7', '8', '9', '+',
                    '/', '=']

    # Check base64 OR codeCheck % 4
    code_fail = [i for i in s if i not in _base64_code]
    if code_fail or len(s) % 4 != 0:
        return False
    return True


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
#         response = dict()
#         if not isBase64(base64_code):
#             response["face_attribute"] = dict()
#             response["face_attribute"]["success"] = False
#             response["face_attribute"]["img_code"] = base64_code
#             response["face_attribute"]["message"] = '图片损坏/非base64编码'
#             response["spendTime"] = "0 s"
#         else:
#             start_time = time.time()
#             try:
#                 image = base64_to_pil(base64_code)
#                 attr_dict = face_attr(image)
#                 stat = True
#             except:
#                 stat = False
#             end_time = time.time()
#
#             response["face_attribute"] = dict()
#             if not stat:
#                 response["face_attribute"]["message"] = '属性提取失败'
#             else:
#                 response["face_attribute"]["message"] = '属性提取成功'
#                 response["face_attribute"]["img_id"] = img_id
#                 response["face_attribute"]["attribute"] = attr_dict
#             response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
#         print(response)
#         self.write(response)


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


# one image inference
def face_attr(img_input):
    celeba_model.eval()
    fairface_model.eval()
    face_mask_model.eval()
    # 图片预处理
    img = transform_test(img_input)
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
    return face_attr_dict(c_output_list, f_output_list, m_output_list)


# one image to loop test
def loop_test(data_path):
    # data_path = 'test_data/aligned_output'
    a = datetime.now()
    img_num = 0
    for file in os.listdir(data_path):
        img_num += 1
        print(file)
        img_path = os.path.join(data_path, file)
        imag = Image.open(img_path).convert('RGB')
        attr_dict = face_attr(imag)
        pprint(attr_dict)
        print("~" * 100)
    b = datetime.now()
    during = (b - a).seconds

    print("batch_size = {}".format(args.batch_size))
    print("num_workers = {}".format(args.num_workers))
    print("image_num = {}".format(img_num))
    print("time = {}".format(during))
    print("infer speed = {}".format(img_num / during))


def interface_test(test_data_path):
    imag = Image.open(test_data_path).convert('RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    print("base64 = {}".format(base64_code))

    response = dict()
    if not isBase64(base64_code):
        response["face_attribute"] = dict()
        response["face_attribute"]["success"] = False
        response["face_attribute"]["img_code"] = base64_code
        response["face_attribute"]["message"] = '图片损坏/非base64编码'
        response["spendTime"] = "0 s"
    else:
        start_time = time.time()
        try:
            image = base64_to_pil(base64_code)
            attr_dict = face_attr(image)
            stat = True
        except:
            stat = False
        end_time = time.time()

        response["face_attribute"] = dict()
        if not stat:
            response["face_attribute"]["message"] = '属性提取失败'
        else:
            response["face_attribute"]["message"] = '属性提取成功'
            response["face_attribute"]["img_id"] = "img_test"
            response["face_attribute"]["attribute"] = attr_dict
        response["spend_time"] = str(round((end_time - start_time), 4) * 1000) + " ms"
    print(response)


if __name__ == '__main__':
    interface_test("test_data/shisuo_face_test/face_于洪-于洪广场家乐福-太湖街_123.305414_41.794836_20210425194134_0.9804584.jpg")
    # loop_test(args.test_data_path)
