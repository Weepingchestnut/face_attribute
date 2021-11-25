import os

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image


def default_loader(path):
    return Image.open(path).convert('RGB')


class MultiLabelDataset(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        images = []
        labels = open(label).readlines()
        for line in labels:
            items = line.split()
            img_name = items.pop(0)
            if os.path.isfile(os.path.join(root, img_name)):
                cur_label = tuple([int(v) for v in items])
                images.append((img_name, cur_label))
            else:
                print(os.path.join(root, img_name) + 'Not Found.')
        self.root = root
        self.images = images
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name, label = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        # raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, torch.Tensor(label)

    def __len__(self):
        return len(self.images)


class get_test_data(data.Dataset):
    def __init__(self, root, label, transform=None, loader=default_loader):
        self.root = root
        self.images = label
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        img_name = self.images[index]
        img = self.loader(os.path.join(self.root, img_name))
        # raw_img = img.copy()
        if self.transform is not None:
            img = self.transform(img)
        return img, img_name

    def __len__(self):
        return len(self.images)


attr_nums = {
    'pa100k': 26,
    'rap': 51,
    'peta': 35,
    'celeba': 10,
    'crop_rapv2': 12,
    'fairface': 17,
    'face_mask': 2,
    'face_attr': 18
}

description = {}
description['pa100k'] = ['Female',
                         'AgeOver60',
                         'Age18-60',
                         'AgeLess18',
                         'Front',
                         'Side',
                         'Back',
                         'Hat',
                         'Glasses',
                         'HandBag',
                         'ShoulderBag',
                         'Backpack',
                         'HoldObjectsInFront',
                         'ShortSleeve',
                         'LongSleeve',
                         'UpperStride',
                         'UpperLogo',
                         'UpperPlaid',
                         'UpperSplice',
                         'LowerStripe',
                         'LowerPattern',
                         'LongCoat',
                         'Trousers',
                         'Shorts',
                         'Skirt&Dress',
                         'boots']

description['peta'] = ['Age16-30',
                       'Age31-45',
                       'Age46-60',
                       'AgeAbove61',
                       'Backpack',
                       'CarryingOther',
                       'Casual lower',
                       'Casual upper',
                       'Formal lower',
                       'Formal upper',
                       'Hat',
                       'Jacket',
                       'Jeans',
                       'Leather Shoes',
                       'Logo',
                       'Long hair',
                       'Male',
                       'Messenger Bag',
                       'Muffler',
                       'No accessory',
                       'No carrying',
                       'Plaid',
                       'PlasticBags',
                       'Sandals',
                       'Shoes',
                       'Shorts',
                       'Short Sleeve',
                       'Skirt',
                       'Sneaker',
                       'Stripes',
                       'Sunglasses',
                       'Trousers',
                       'Tshirt',
                       'UpperOther',
                       'V-Neck']

description['rap'] = ['Female',
                      'AgeLess16',
                      'Age17-30',
                      'Age31-45',
                      'BodyFat',
                      'BodyNormal',
                      'BodyThin',
                      'Customer',
                      'Clerk',
                      'BaldHead',
                      'LongHair',
                      'BlackHair',
                      'Hat',
                      'Glasses',
                      'Muffler',
                      'Shirt',
                      'Sweater',
                      'Vest',
                      'TShirt',
                      'Cotton',
                      'Jacket',
                      'Suit-Up',
                      'Tight',
                      'ShortSleeve',
                      'LongTrousers',
                      'Skirt',
                      'ShortSkirt',
                      'Dress',
                      'Jeans',
                      'TightTrousers',
                      'LeatherShoes',
                      'SportShoes',
                      'Boots',
                      'ClothShoes',
                      'CasualShoes',
                      'Backpack',
                      'SSBag',
                      'HandBag',
                      'Box',
                      'PlasticBag',
                      'PaperBag',
                      'HandTrunk',
                      'OtherAttchment',
                      'Calling',
                      'Talking',
                      'Gathering',
                      'Holding',
                      'Pusing',
                      'Pulling',
                      'CarryingbyArm',
                      'CarryingbyHand']

description['celeba'] = [
    '性别',   # 0     (1男 0女)
    '秃头',   # 1
    '直发',   # 2
    '卷发',   # 3
    '黑发',   # 4
    '金发',   # 5
    '棕发',   # 6
    '灰白发',  # 7
    '眼镜',   # 8
    '帽子'    # 9
]

description['crop_rapv2'] = [
    '性别',   # 0     (1女 0男)
    '16-',      # 1
    '17-30',    # 2
    '31-45',    # 3
    '46-60',    # 4
    '60+',      # 5
    '光头',       # 6
    '长发',       # 7
    '黑发',       # 8
    '戴帽',       # 9
    '眼镜',       # 10
    '太阳镜'       # 11
]

description['fairface'] = [
    '性别',       # 0     (1男 0女)
    '0-2',      # 1
    '3-9',      # 2
    '10-19',    # 3
    '20-29',    # 4
    '30-39',    # 5
    '40-49',    # 6
    '50-59',    # 7
    '60-69',    # 8
    '70+',      # 9
    '白种人',            # 10    白种人
    '黑种人',            # 11    黑种人
    '拉美裔西班牙人',  # 12    拉美裔西班牙人
    '东亚黄种人',       # 13    东亚黄种人
    '东南亚黄种人',  # 14    东南亚黄种人
    '印第安人',           # 15    印第安人
    '中东人'    # 16    中东人
]

description['face_mask'] = [
    'Mask',
    'NoMask'
]

description['face_attr'] = [
    '性别',   # (1男 0女)       0
    '少年（20以下）',         # 1
    '青年（20-40）',        # 2
    '中年（40-60）',        # 3
    '老年（60以上）',         # 4
    '白种人',              # 5
    '黑种人',              # 6
    '黄种人',              # 7
    '秃头',               # 8
    '直发',               # 9
    '卷发',               # 10
    '黑发',               # 11
    '金发',               # 12
    '棕发',               # 13
    '灰白发',              # 14
    '是否戴眼镜',            # 15
    '是否戴帽子',            # 16
    '是否戴口罩'             # 17
]


def Get_Dataset(experiment, approach):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform_train = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=30, translate=(0.3, 0.3), fillcolor=(0, 0, 0)),
        # transforms.ColorJitter(hue=.05, saturation=.05),
        # transforms.RandomRotation(20, resample=Image.BILINEAR),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(size=(128, 128)),
        transforms.ToTensor(),
        normalize
    ])

    if experiment == 'pa100k':
        train_dataset = MultiLabelDataset(root='data_path',
                                          label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                                        label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['pa100k'], description['pa100k']
    elif experiment == 'rap':
        train_dataset = MultiLabelDataset(root='data_path',
                                          label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                                        label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['rap'], description['rap']
    elif experiment == 'peta':
        train_dataset = MultiLabelDataset(root='data_path',
                                          label='train_list_path', transform=transform_train)
        val_dataset = MultiLabelDataset(root='data_path',
                                        label='val_list_path', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['peta'], description['peta']
    elif experiment == 'celeba':
        train_dataset = MultiLabelDataset(root='face_data/CelebA/Img',
                                          label='data_list/celeba/need_attr_list.txt', transform=transform_train)
        val_dataset = MultiLabelDataset(root='face_data/CelebA/Img',
                                        label='data_list/celeba/test.txt', transform=transform_test)
        return train_dataset, val_dataset, attr_nums['celeba'], description['celeba']
    elif experiment == 'crop_rapv2':
        train_dataset = MultiLabelDataset(root='face_data/crop_rapv2',
                                          label='data_list/crop_rapv2/train_list.txt',
                                          transform=transform_train)
        val_dataset = MultiLabelDataset(root='face_data/crop_rapv2',
                                        label='data_list/crop_rapv2/test_list.txt',
                                        transform=transform_test)
        return train_dataset, val_dataset, attr_nums['crop_rapv2'], description['crop_rapv2']
    elif experiment == 'fairface':
        train_dataset = MultiLabelDataset(root='face_data/FairFace/train',
                                          label='data_list/fairface/fairface_label_train.txt',
                                          transform=transform_train)
        val_dataset = MultiLabelDataset(root='face_data/FairFace/val',
                                        label='data_list/fairface/fairface_label_val.txt',
                                        transform=transform_test)
        return train_dataset, val_dataset, attr_nums['fairface'], description['fairface']
    elif experiment == 'face_mask':
        train_dataset = MultiLabelDataset(root='face_data/face_mask/Img',
                                          label='data_list/face_mask/train.txt',
                                          transform=transform_train)
        val_dataset = MultiLabelDataset(root='face_data/face_mask/Img',
                                        label='data_list/face_mask/test.txt',
                                        transform=transform_test)
        return train_dataset, val_dataset, attr_nums['face_mask'], description['face_mask']
