import argparse
import warnings
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.backends import cudnn

import model as models
from utils.datasets import attr_nums, get_test_data
from utils.display import *

warnings.filterwarnings('ignore')
# sys.path.append('model/')

parser = argparse.ArgumentParser(description='Face Attribute Framework')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--test_data_path', default='test_data/shisuo_face_test', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('-s', '--show', dest='show', action='store_true', required=False,
                    help='show attribute in imag')
parser.add_argument('-c', '--confidence', dest='confidence', action='store_true', required=False,
                    help='print attribute confidence in imag')
parser.add_argument('--save_path', default='work_dir/shisuo_face_test_output_img', type=str, required=False,
                    help='(default=%(default)s)')

args = parser.parse_args()

resume_path = {
    # 'celeba': 'checkpoint/3_ce_psize128_mA_89-19.pth.tar',
    'celeba': 'checkpoint/4_ce_mA89-72.pth.tar',
    # 'fairface': 'checkpoint/11_fair_psize128_76-62.pth.tar',
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


def batch_test(test_data_path):
    # make img_name list
    img_names = []
    for file in os.listdir(test_data_path):
        # 判断是否为空图像，跳过空图像，防止后续读取图像是报错
        if os.stat(os.path.join(test_data_path, file)).st_size == 0:
            pass
        else:
            img_names.append(file)

    test_dataset = get_test_data(root=test_data_path, label=img_names, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    a = datetime.now()
    test(test_loader, celeba_model, fairface_model, face_mask_model)
    b = datetime.now()
    during = (b - a).seconds
    print("batch_size = {}".format(args.batch_size))
    print("num_workers = {}".format(args.num_workers))
    print("image_num = {} 张".format(test_dataset.__len__()))
    print("time = {} s".format(during))
    try:
        print("infer speed = {} 张/s".format(round(test_dataset.__len__() / during, 2)))
    except ZeroDivisionError:
        print("推理时间不足1s")


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
            if args.confidence:
                attr_dict = face_attr_dict_c(c_output_list, f_output_list, m_output_list)
            else:
                attr_dict = face_attr_dict(c_output_list, f_output_list, m_output_list)
            # pprint(face_attr_dict(c_output_list, f_output_list, m_output_list))
            pprint(attr_dict)
            if args.show:
                show_attribute_img(one_img_name, attr_dict)


if __name__ == '__main__':
    batch_test(args.test_data_path)
