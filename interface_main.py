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

import torch.utils.data as data
import model as models
from utils.datasets import attr_nums
from utils.display import *

warnings.filterwarnings('ignore')
# sys.path.append('model/')

parser = argparse.ArgumentParser(description='Face Attribute Framework')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=0, type=int, required=False, help='(default=%(default)d)')

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


def res_message(img_index, mode):
    if mode == 'id':
        return '第' + str(img_index) + '张图片id为空'
    elif mode == 'base64':
        return '第' + str(img_index) + '张图片损坏/非base64编码'
    elif mode == 'dict_format_0':
        return '第' + str(img_index) + '张图片JSON格式缺失键img_id与base64_code'
    elif mode == 'dict_format_1':
        return '第' + str(img_index) + '张图片JSON缺失键base64_code'
    elif mode == 'dict_format_10':
        return '第' + str(img_index) + '张图片JSON缺失键img_id'


def is_need_dict(img_dict):
    record = 0
    if 'img_id' in img_dict.keys():
        record = record + 1
    if 'base64_code' in img_dict.keys():
        record = record + 10
    return record
    # return 'img_id' in img_dict.keys() and 'base64_code' in img_dict.keys()


class base64_api(tornado.web.RequestHandler):
    def initialize(self, gconf):
        self.config = gconf
        self.pool = gconf.get("threadpool", None)

    # @tornado.web.asynchronous
    @tornado.gen.coroutine
    def post(self, *aegs, **kwargs):
        start_time = time.time()
        request = json.loads(self.request.body)
        # img_id = request.get('img_id', '')
        # base64_code = request.get('base64_code', '')
        img_list = request.get('img_list', '')

        test_dataset = get_interface_data(imgs_list=img_list)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
        )

        stat = True
        response = {'success': True, 'message': ['属性提取成功'], 'attribute': [], 'spendTime': '0 ms'}
        for i, img_dict in enumerate(img_list):
            # 判断JSON是否满足dict格式要求
            get_record = is_need_dict(img_dict)
            if get_record < 11:
                response['success'] = False
                response['message'][0] = '属性提取失败'
                response['message'].append(res_message(i + 1, 'dict_format_' + str(get_record)))
                stat = False
            else:
                # 图片id为空（字符串长度为0 or 字符串均为空格）
                if len(str(img_dict['img_id'])) == 0 or str(img_dict['img_id']).isspace is True:
                    response['success'] = False
                    response['message'][0] = '属性提取失败'
                    response['message'].append(res_message(i + 1, 'id'))
                    stat = False

                if not isBase64(img_dict['base64_code']):
                    response['success'] = False
                    response['message'][0] = '属性提取失败'
                    response['message'].append(res_message(i + 1, 'base64'))
                    stat = False

        if not stat:
            pass
        else:
            # start_time = time.time()
            response['attribute'] = test(test_loader, celeba_model, fairface_model, face_mask_model)
            # end_time = time.time()
            # print(str(round((end_time - start_time), 4) * 1000) + " ms")
        end_time = time.time()
        response['spendTime'] = str(round((end_time - start_time), 4) * 1000) + " ms"
        self.write(response)
        print(response)


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


class get_interface_data(data.Dataset):
    def __init__(self, imgs_list, transform=transform_test, loader=base64_to_pil):
        self.images = imgs_list
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_id = self.images[index]['img_id']
        img = self.loader(self.images[index]['base64_code'])

        if self.transform is not None:
            img = self.transform(img)
        return img, img_id

    def __len__(self):
        return len(self.images)


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


def interface_test():
    imag = Image.open(
        'test_data/shisuo_face_test/face_于洪-4G清水湾温泉_123.296744_41.79479_20210416104413_0.9802733.jpg').convert('RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    # print("base64 = {}".format(base64_code))
    img1 = {
        'img_id': 'face_于洪-4G清水湾温泉_123.296744_41.79479_20210416104413_0.9802733.jpg',
        'base64_code': base64_code
    }

    imag = Image.open(
        'test_data/shisuo_face_test/face_于洪-ZH西湖叠院东门人行出_123.193184_41.799646_20210416113518_0.9816.jpg').convert('RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    # print("base64 = {}".format(base64_code))
    img2 = {
        'img_id': 'face_于洪-ZH西湖叠院东门人行出_123.193184_41.799646_20210416113518_0.9816.jpg',
        'base64_code': base64_code
    }

    imag = Image.open(
        'test_data/shisuo_face_test/face_于洪-于洪广场家乐福-太湖街_123.305414_41.794836_20210416113449_0.99417347.jpg').convert(
        'RGB')
    base64_code = str(pil_to_base64(imag), 'utf-8')  # base64编码b' '前缀的去除
    # print("base64 = {}".format(base64_code))
    img3 = {
        '': 'face_于洪-于洪广场家乐福-太湖街_123.305414_41.794836_20210416113449_0.99417347.jpg',
        'base64_code': base64_code
    }

    info_list = [img1, img2, img3]
    # pprint(info_list)

    test_dataset = get_interface_data(imgs_list=info_list)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    stat = True
    response = {'success': True, 'message': ['属性提取成功'], 'attribute': [], 'spendTime': '0 ms'}
    for i, img_dict in enumerate(info_list):
        # 判断JSON是否满足dict格式要求
        get_record = is_need_dict(img_dict)
        if get_record < 11:
            response['success'] = False
            response['message'][0] = '属性提取失败'
            response['message'].append(res_message(i + 1, 'dict_format_' + str(get_record)))
            stat = False
        else:
            # 图片id为空（字符串长度为0 or 字符串均为空格）
            if len(str(img_dict['img_id'])) == 0 or str(img_dict['img_id']).isspace is True:
                response['success'] = False
                response['message'][0] = '属性提取失败'
                response['message'].append(res_message(i + 1, 'id'))
                stat = False

            if not isBase64(img_dict['base64_code']):
                response['success'] = False
                response['message'][0] = '属性提取失败'
                response['message'].append(res_message(i + 1, 'base64'))
                stat = False

    if not stat:
        # print("stat = False")
        pass
    else:
        # print("stat = True")
        start_time = time.time()
        response['attribute'] = test(test_loader, celeba_model, fairface_model, face_mask_model)
        end_time = time.time()
        # print(str(round((end_time - start_time), 4) * 1000) + " ms")
        # response['message'].append('属性提取成功')
        response['spendTime'] = str(round((end_time - start_time), 4) * 1000) + " ms"

    print(response)


def test(test_loader, c_model, f_model, m_model):
    c_model.eval()
    f_model.eval()
    m_model.eval()
    attr = []
    # img_dict = {'img_id': '', 'attr_dict': dict()}

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
            # img_dict = {'img_id': '', 'attr_dict': dict()}
            img_dict = dict()
            # print("img_name: {}".format(img_name[one_bs]))
            one_img_name = img_name[one_bs]
            c_output_list = c_output[one_bs].tolist()
            f_output_list = f_output[one_bs].tolist()
            m_output_list = m_output[one_bs].tolist()

            # 置信度微调
            if max(f_output_list[10], f_output_list[11], f_output_list[12], f_output_list[14],
                   f_output_list[15], f_output_list[16]) > 0.8:
                pass
            else:
                f_output_list[13] = f_output_list[13] * 1.5

            attr_dict = face_attr_dict(c_output_list, f_output_list, m_output_list)

            img_dict['img_id'] = one_img_name
            img_dict['attr_dict'] = attr_dict
            attr.append(img_dict)
            # pprint(attr_dict)
    return attr


if __name__ == '__main__':
    interface_test()
