import os
from pprint import pprint

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from infer_main import args
from utils.attri_dict import *
from utils.datasets import description

data_path = args.test_data_path
save_path = args.save_path

if not os.path.exists(save_path):
    os.mkdir(save_path)


def show_attribute_img(img_name, attr_dict):
    img_path = os.path.join(data_path, img_name)
    print("img_path = {}".format(img_path))
    attr_list = list(attr_dict.keys())
    # print(attr_list)

    img = cv2.imread(img_path)
    if isinstance(img, np.ndarray):
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    # 字体格式设置
    font_style = ImageFont.truetype("./checkpoint/msht.ttf", size=10, encoding="utf-8")

    for index, attr in enumerate(attr_list):
        draw.text((0, 0 + index * 15), attr + ": " + attr_dict[attr], (178, 34, 34), font_style)

    img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    cv2.imwrite(os.path.join(save_path, img_name), img)
    print("save_path = {}".format(os.path.join(save_path, img_name)))
    print('~' * 100)


def face_attr_dict(c_output_list, f_output_list, m_output_list):
    # c_display_dict = dict(zip(description['celeba'], c_output_list))
    # f_display_dict = dict(zip(description['fairface'], f_output_list))
    # m_display_dict = dict(zip(description['face_mask'], m_output_list))
    # pprint(f_display_dict)
    # pprint(c_display_dict)
    # pprint(m_display_dict)

    face_dict = {
        '性别': '未知',
        '年龄': '未知',
        '种族': '未知',
        '发型': '未知',
        '发色': '未知',
        '是否戴眼镜': '未知',
        '是否戴帽子': '未知',
        '是否戴口罩': '未知'
    }
    # c_gender = c_output_list[0]
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

    # if c_gender > 0.5:
    #     face_dict['性别'] = gender_dict[0] + "(celeba) "    # male
    # else:
    #     face_dict['性别'] = gender_dict[1] + "(celeba) "   # female
    # if format(f_gender, '.1f') == 0.5:
    if abs(f_gender - 0.5) < 0.1:  # 0.4~0.6之间未知
        pass
    elif f_gender > 0.5:
        face_dict['性别'] = gender_dict[0]  # 男
    else:
        face_dict['性别'] = gender_dict[1]  # 女

    face_dict['年龄'] = age_dict[age.index(max(age))]

    if max(race) < 0.5:     # 未知
        pass
    else:
        race_index = race.index(max(race))
        face_dict['种族'] = race_dict[race_index]

    if max(hair_style) < 0.1:
        pass
    else:
        hair_style_index = hair_style.index(max(hair_style))
        face_dict['发型'] = hair_style_dict[hair_style_index]

    if max(hair_color) < 0.1:
        pass
    else:
        hair_color_index = hair_color.index(max(hair_color))
        face_dict['发色'] = hair_color_dict[hair_color_index]

    if glasses > 0.1:
        face_dict['是否戴眼镜'] = is_dict[0]
    else:
        face_dict['是否戴眼镜'] = is_dict[1]

    if hat > 0.5:
        face_dict['是否戴帽子'] = is_dict[0]
    else:
        face_dict['是否戴帽子'] = is_dict[1]

    mask_index = m_output_list.index(max(m_output_list))
    face_dict['是否戴口罩'] = is_dict[mask_index]

    return face_dict


def face_attr_dict_F1(c_output_list, f_output_list, m_output_list):
    # c_display_dict = dict(zip(description['celeba'], c_output_list))
    # f_display_dict = dict(zip(description['fairface'], f_output_list))
    # m_display_dict = dict(zip(description['face_mask'], m_output_list))
    # pprint(f_display_dict)
    # pprint(c_display_dict)
    # pprint(m_display_dict)

    face_dict = {
        '性别': '未知',
        '年龄': '未知',
        '种族': '未知',
        '发型': '未知',
        '发色': '未知',
        '是否戴眼镜': '未知',
        '是否戴帽子': '未知',
        '是否戴口罩': '未知'
    }
    # c_gender = c_output_list[0]
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

    F1_list = [0] * 18
    # if c_gender > 0.5:
    #     face_dict['性别'] = gender_dict[0] + "(celeba) "    # male
    # else:
    #     face_dict['性别'] = gender_dict[1] + "(celeba) "   # female
    # if format(f_gender, '.1f') == 0.5:
    if abs(f_gender - 0.5) < 0.1:  # 0.4~0.6之间未知
        F1_list[0] = -1     # -1 表“未知”
    elif f_gender > 0.5:
        face_dict['性别'] = gender_dict[0]  # 男
        F1_list[0] = 1
    else:
        face_dict['性别'] = gender_dict[1]  # 女

    face_dict['年龄'] = age_dict[age.index(max(age))]
    F1_list[age.index(max(age)) + 1] = 1

    if max(race) < 0.5:     # 未知
        F1_list[5] = -1
        F1_list[6] = -1
        F1_list[7] = -1
    else:
        race_index = race.index(max(race))
        face_dict['种族'] = race_dict[race_index]
        if race_index == 2 or race_index == 5:
            F1_list[5] = -1
            F1_list[6] = -1
            F1_list[7] = -1
        elif race_index == 0 or race_index == 6:   # 白种人
            F1_list[5] = 1
        elif race_index == 1:   # 黑种人
            F1_list[6] = 1
        elif race_index == 3 or race_index == 4:    # 黄种人
            F1_list[7] = 1

    if max(hair_style) < 0.1:
        F1_list[8] = -1
        F1_list[9] = -1
        F1_list[10] = -1
    else:
        hair_style_index = hair_style.index(max(hair_style))
        face_dict['发型'] = hair_style_dict[hair_style_index]
        if hair_style_index == 0:   # 秃头
            F1_list[8] = 1
        elif hair_style_index == 1:     # 直发
            F1_list[9] = 1
        elif hair_style_index == 2:     # 卷发
            F1_list[10] = 1

    if max(hair_color) < 0.1:
        F1_list[11] = -1
        F1_list[12] = -1
        F1_list[13] = -1
        F1_list[14] = -1
    else:
        hair_color_index = hair_color.index(max(hair_color))
        face_dict['发色'] = hair_color_dict[hair_color_index]
        if hair_color_index == 0:   # 黑发
            F1_list[11] = 1
        elif hair_color_index == 1:     # 金发
            F1_list[12] = 1
        elif hair_color_index == 2:     # 棕发
            F1_list[13] = 1
        elif hair_color_index == 3:     # 灰白发
            F1_list[14] = 1

    if glasses > 0.1:
        face_dict['是否戴眼镜'] = is_dict[0]
        F1_list[15] = 1
    else:
        face_dict['是否戴眼镜'] = is_dict[1]

    if hat > 0.5:
        face_dict['是否戴帽子'] = is_dict[0]
        F1_list[16] = 1
    else:
        face_dict['是否戴帽子'] = is_dict[1]

    mask_index = m_output_list.index(max(m_output_list))
    face_dict['是否戴口罩'] = is_dict[mask_index]
    if mask_index == 0:     # 戴口罩
        F1_list[17] = 1
    print("F1_list = {}".format(F1_list))

    return F1_list


# print confidence
def face_attr_dict_c(c_output_list, f_output_list, m_output_list):
    # c_display_dict = dict(zip(description['celeba'], c_output_list))
    # f_display_dict = dict(zip(description['fairface'], f_output_list))
    # m_display_dict = dict(zip(description['face_mask'], m_output_list))
    # pprint(f_display_dict)
    # pprint(c_display_dict)
    # pprint(m_display_dict)

    face_dict = {
        '性别': '未知',
        '年龄': '未知',
        '种族': '未知',
        '发型': '未知',
        '发色': '未知',
        '是否戴眼镜': '未知',
        '是否戴帽子': '未知',
        '是否戴口罩': '未知'
    }
    # c_gender = c_output_list[0]
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

    # if c_gender > 0.5:
    #     face_dict['性别'] = gender_dict[0] + "(celeba) "    # male
    # else:
    #     face_dict['性别'] = gender_dict[1] + "(celeba) "   # female
    # if format(f_gender, '.1f') == 0.5:
    if abs(f_gender - 0.5) < 0.1:   # 0.4~0.6之间未知
        face_dict['性别'] = face_dict['性别'] + format(f_gender, '.4f')
        # pass
    elif f_gender > 0.5:
        face_dict['性别'] = gender_dict[0] + format(f_gender, '.4f')    # + "(fairface)"
    else:
        face_dict['性别'] = gender_dict[1] + format(f_gender, '.4f')   # + "(fairface)"

    face_dict['年龄'] = age_dict[age.index(max(age))] + format(max(age), '.4f')

    if max(race) < 0.5:
        face_dict['种族'] = face_dict['种族'] + ' ' + race_dict_[race.index(max(race))] + format(max(race), '.4f')
        # pass
    else:
        face_dict['种族'] = race_dict[race.index(max(race))] + "({})".format(race_dict_[race.index(max(race))]) + format(max(race), '.4f')

    if max(hair_style) < 0.1:
        face_dict['发型'] = face_dict['发型'] + ' ' + hair_style_dict[hair_style.index(max(hair_style))] + format(max(hair_style), '.4f')
        pass
    else:
        face_dict['发型'] = hair_style_dict[hair_style.index(max(hair_style))] + format(max(hair_style), '.4f')

    if max(hair_color) < 0.1:
        face_dict['发色'] = face_dict['发色'] + ' ' + hair_color_dict[hair_color.index(max(hair_color))] + format(max(hair_color), '.4f')
    else:
        face_dict['发色'] = hair_color_dict[hair_color.index(max(hair_color))] + format(max(hair_color), '.4f')

    if glasses > 0.1:
        face_dict['是否戴眼镜'] = is_dict[0] + format(glasses, '.4f')
    else:
        face_dict['是否戴眼镜'] = is_dict[1] + format(glasses, '.4f')

    if hat > 0.5:
        face_dict['是否戴帽子'] = is_dict[0] + format(hat, '.4f')
    else:
        face_dict['是否戴帽子'] = is_dict[1] + format(hat, '.4f')

    if m_output_list.index(max(m_output_list)) == 0:    # 是
        face_dict['是否戴口罩'] = is_dict[m_output_list.index(max(m_output_list))] + format(max(m_output_list), '.4f')
    else:
        # 否 置信度调整到(0, 1)，否 --> 0；是 --> 1
        face_dict['是否戴口罩'] = is_dict[m_output_list.index(max(m_output_list))] + format(1 - max(m_output_list), '.4f')

    return face_dict


def celeba_attr_dict(output_list):
    display_dict = dict(zip(description['celeba'], output_list))
    pprint(display_dict)

    face_dict = {'性别': '',
                 '发型': '',
                 '发色': '',
                 '是否戴眼镜': '',
                 '是否戴帽子': ''}
    gender = output_list[0]
    hair_style = [output_list[1], output_list[2], output_list[3]]
    hair_color = [output_list[4], output_list[5], output_list[6], output_list[7]]
    glasses = output_list[8]
    hat = output_list[9]
    # print("hat = {}".format(hat))

    if gender > 0.5:
        face_dict['性别'] = '男'
    else:
        face_dict['性别'] = '女'

    face_dict['发型'] = hair_style_dict[hair_style.index(max(hair_style))]
    face_dict['发色'] = hair_color_dict[hair_color.index(max(hair_color))]

    if glasses > .5:
        face_dict['是否戴眼镜'] = is_dict[0]
    else:
        face_dict['是否戴眼镜'] = is_dict[1]
    if hat > .5:
        face_dict['是否戴帽子'] = is_dict[0]
    else:
        face_dict['是否戴帽子'] = is_dict[1]

    return face_dict


def crop_rapv2_attr_dict(output_list):
    display_dict = dict(zip(description['crop_rapv2'], output_list))
    pprint(display_dict)
    face_dict = {'性别': '',
                 '年龄': '',
                 '发型': '',
                 '发色': '',
                 '是否戴帽子': '',
                 '是否戴眼镜': ''}

    gender = output_list[0]
    age = [output_list[1], output_list[2], output_list[3], output_list[4], output_list[5]]
    hair_style = [output_list[6], output_list[7]]
    hair_color = output_list[8]
    hat = output_list[9]
    glasses = [output_list[10], output_list[11]]

    if gender > .5:
        face_dict['性别'] = gender_dict[1]  # female
    else:
        face_dict['性别'] = gender_dict[0]  # male

    face_dict['年龄'] = age_dict_v2[age.index(max(age))]

    if max(hair_style) > .5:
        face_dict['发型'] = hair_style_dict_rap[hair_style.index(max(hair_style))]
    else:
        face_dict['发型'] = hair_style_dict_rap[2]

    if hair_color > .5:
        face_dict['发色'] = hair_color_dict_rap[0]
    else:
        face_dict['发色'] = hair_color_dict_rap[1]

    if hat > .5:
        face_dict['是否戴帽子'] = is_dict[0]
    else:
        face_dict['是否戴帽子'] = is_dict[1]

    if max(glasses) > .5:
        face_dict['是否戴眼镜'] = is_dict[0]
    else:
        face_dict['是否戴眼镜'] = is_dict[1]

    return face_dict


def fairface_attr_dict(output_list):
    display_dict = dict(zip(description['fairface'], output_list))
    pprint(display_dict)
    face_dict = {'性别': '',
                 '年龄': '',
                 '种族': ''}

    gender = output_list[0]
    age = [output_list[1], output_list[2], output_list[3], output_list[4], output_list[5],
           output_list[6], output_list[7], output_list[8], output_list[9]]
    race = [output_list[10], output_list[11], output_list[12], output_list[13],
            output_list[14], output_list[15], output_list[16]]

    if gender > .5:
        face_dict['性别'] = gender_dict[0]  # 男
    else:
        face_dict['性别'] = gender_dict[1]  # 女
    face_dict['年龄'] = age_dict_fair[age.index(max(age))]
    face_dict['种族'] = race_dict[race.index(max(race))]

    return face_dict


def mask_face_attr_dict(output_list):
    display_dict = dict(zip(description['face_mask'], output_list))
    pprint(display_dict)
    face_dict = {'是否戴口罩': id2class[output_list.index(max(output_list))]}

    return face_dict


if __name__ == '__main__':
    F1 = [0]*18
    print(F1)


