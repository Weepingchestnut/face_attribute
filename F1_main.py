import argparse
import warnings
from datetime import datetime

import torch
import torchvision.transforms as transforms
from torch.backends import cudnn
from tqdm import tqdm

import model as models
from utils.datasets import attr_nums, MultiLabelDataset
from utils.display import *

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser(description='Face Attribute Framework')
parser.add_argument('--batch_size', default=128, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--num_workers', default=4, type=int, required=False, help='(default=%(default)d)')
parser.add_argument('--test_data_path', default='test_data/F1_test', type=str, required=False,
                    help='(default=%(default)s)')
parser.add_argument('--test_data_label', default='test_data/F1_main_test.txt', type=str, required=False,
                    help='(default=%(default)s)')

args = parser.parse_args()

resume_path = {
    'celeba': 'checkpoint/4_ce_mA89-72.pth.tar',
    'fairface': 'checkpoint/fairface_36_76-66.pth.tar',
    'face_mask': 'checkpoint/2_fm_psize128_mA_99-99.pth.tar'
}


def Get_TestDataset(data_path, data_label):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_test = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        normalize
    ])
    test_dataset = MultiLabelDataset(root=data_path, label=data_label, transform=transform_test)

    return test_dataset, attr_nums['face_attr'], description['face_attr']


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


def F1_test():
    test_dataset, attr_num, description = Get_TestDataset(args.test_data_path, args.test_data_label)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )

    a = datetime.now()
    test(test_loader, celeba_model, fairface_model, face_mask_model, attr_num, description)
    b = datetime.now()
    during = (b - a).seconds
    print("batch_size = {}".format(args.batch_size))
    print("num_workers = {}".format(args.num_workers))
    print("image_num = {} 张".format(test_dataset.__len__()))
    print("time = {} s".format(during))
    try:
        print("infer speed = {} 张/s".format(test_dataset.__len__() / during))
    except ZeroDivisionError:
        print("推理时间不足1s")


def test(val_loader, c_model, f_model, m_model, attr_num, description):
    c_model.eval()
    f_model.eval()
    m_model.eval()

    pos_cnt = []
    pos_tol = []
    neg_cnt = []
    neg_tol = []

    accu = 0.0
    prec = 0.0
    recall = 0.0
    tol = 0

    for it in range(attr_num):
        pos_cnt.append(0)
        pos_tol.append(0)
        neg_cnt.append(0)
        neg_tol.append(0)

    for i, _ in tqdm(enumerate(val_loader), total=len(val_loader)):
        input, target = _
        input = input.cuda(non_blocking=True)
        c_output, f_output, m_output = c_model(input), f_model(input), m_model(input)
        # bs = input.size(0)

        # maximum voting
        c_output = torch.max(torch.max(torch.max(c_output[0], c_output[1]), c_output[2]), c_output[3])
        f_output = torch.max(torch.max(torch.max(f_output[0], f_output[1]), f_output[2]), f_output[3])
        m_output = torch.max(torch.max(torch.max(m_output[0], m_output[1]), m_output[2]), m_output[3])

        batch_size = target.size(0)
        tol = tol + batch_size
        c_output = torch.sigmoid(c_output.data).cpu().numpy()   # [128, ]
        f_output = torch.sigmoid(f_output.data).cpu().numpy()
        m_output = torch.sigmoid(m_output.data).cpu().numpy()
        output = []

        for one_bs in range(batch_size):
            # print("img_name: {}".format(img_name[one_bs]))
            # one_img_name = img_name[one_bs]
            c_output_list = c_output[one_bs].tolist()
            f_output_list = f_output[one_bs].tolist()
            m_output_list = m_output[one_bs].tolist()

            # 置信度微调
            if max(f_output_list[10], f_output_list[11], f_output_list[12], f_output_list[14],
                   f_output_list[15], f_output_list[16]) > 0.8:
                pass
            else:
                f_output_list[13] = f_output_list[13] * 1.5

            one_bs_output = face_attr_dict_F1(c_output_list, f_output_list, m_output_list)
            output.append(one_bs_output)

        output = np.array(output)
        # output = np.where(output > 0.5, 1, 0)
        target = target.cpu().numpy()

        for it in range(attr_num):
            for jt in range(batch_size):
                if target[jt][it] == 1:
                    pos_tol[it] = pos_tol[it] + 1
                    if output[jt][it] == 1:
                        pos_cnt[it] = pos_cnt[it] + 1
                if target[jt][it] == 0:
                    neg_tol[it] = neg_tol[it] + 1
                    if output[jt][it] == 0:
                        neg_cnt[it] = neg_cnt[it] + 1

        if attr_num == 1:
            continue
        for jt in range(batch_size):
            tp = 0
            fn = 0
            fp = 0
            for it in range(attr_num):
                if (output[jt][it] == 1 and target[jt][it] == 1) or (output[jt][it] == -1 and target[jt][it] == 1):
                    tp = tp + 1
                elif output[jt][it] == 0 and target[jt][it] == 1:
                    fn = fn + 1
                elif output[jt][it] == 1 and target[jt][it] == 0:
                    fp = fp + 1
            if tp + fn + fp != 0:
                accu = accu + 1.0 * tp / (tp + fn + fp)
            if tp + fp != 0:
                prec = prec + 1.0 * tp / (tp + fp)
            if tp + fn != 0:
                recall = recall + 1.0 * tp / (tp + fn)

    print('=' * 100)
    print('\t     Attr              \tp_true/n_true\tp_tol/n_tol\tp_pred/n_pred\tcur_mA')
    mA = 0.0
    for it in range(attr_num):
        # print("it = {}, description[it] = {}".format(it, description[it]))
        cur_mA = ((1.0 * pos_cnt[it] / pos_tol[it]) + (1.0 * neg_cnt[it] / neg_tol[it])) / 2.0
        mA = mA + cur_mA
        print('\t#{:2}: {:18}\t{:4}\{:4}\t{:4}\{:4}\t{:4}\{:4}\t{:.5f}'.format(it,
                                                                               description[it],
                                                                               pos_cnt[it],
                                                                               neg_cnt[it],
                                                                               pos_tol[it],
                                                                               neg_tol[it],
                                                                               (pos_cnt[it] + neg_tol[it] - neg_cnt[it]),
                                                                               (neg_cnt[it] + pos_tol[it] - pos_cnt[it]),
                                                                               cur_mA))
    mA = mA / attr_num
    print('\t' + 'mA:        ' + str(mA))

    if attr_num != 1:
        accu = accu / tol
        prec = prec / tol
        recall = recall / tol
        f1 = 2.0 * prec * recall / (prec + recall)
        print('\t' + 'Accuracy:  ' + str(accu))
        print('\t' + 'Precision: ' + str(prec))
        print('\t' + 'Recall:    ' + str(recall))
        print('\t' + 'F1_Score:  ' + str(f1))
    print('=' * 100)


if __name__ == '__main__':
    F1_test()
