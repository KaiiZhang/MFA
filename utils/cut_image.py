import itertools
from skimage.io import imread, imsave
import numpy as np
import os
# from tqdm import tqdm
from PIL import Image

cityspallete = [
    128, 64, 128,
    244, 35, 232,
    70, 70, 70,
    102, 102, 156,
    190, 153, 153,
    153, 153, 153,
    250, 170, 30,
    220, 220, 0,
    107, 142, 35,
    152, 251, 152,
    0, 130, 180,
    220, 20, 60,
    255, 0, 0,
    0, 0, 142,
    0, 0, 70,
    0, 60, 100,
    0, 0, 0,
]

id_to_trainid = {7:18 ,8: 7, 9: 8, 10: 9, 11: 10, 12: 11, 13: 12,
                              14: 13, 15: 14, 16: 15, 17: 16, 18:17}
id_to_name = ['旱地', '水田', '林地', '草地', '建设用地', '道路', '河流水面', '湖泊水面', '水库坑塘', '潮间带',
              '草本沼泽', '沼泽地', '红树林地', '盐田', '养殖', '浅海', '其他土地']

def get_color_pallete(npimg):
    """Visualize image.
    Parameters
    ----------
    npimg : numpy.ndarray
        Single channel image with shape `H, W`.
    dataset : str, default: 'pascal_voc'
        The dataset that model pretrained on. ('pascal_voc', 'ade20k')
    Returns
    -------
    out_img : PIL.Image
        Image with color pallete
    """
    out_img = Image.fromarray(np.uint8(npimg))
    out_img.putpalette(cityspallete)
    return out_img

def normalize_for_IGSNRR(img, max_pixel_values=[7883.5, 6334.5, 5642.75, 7827.75, 0.7721173167228699, 0.561278760433197]):
    channel_num = img.shape[-1]
    max_pixel_values = max_pixel_values[:channel_num]
    max_pixel_values = np.array(max_pixel_values)
    img = img.astype(np.float32)
    img /= max_pixel_values
    img *= 255.0
    return img

def img_cut_spcific_num(read_path, gt_path = None, imgcHeight=256, imgcWidth=256):
    '''
    剪切原始较大的图片和标签到指定的大小尺寸
    对原始图片做了通道变换 从bgrn...变换到rgbn...
    对元素值做了缩放，归一化到0-255
    img float
    gt uint8
    '''
    im = imread(read_path)
    im[...,:3] = im[...,:3][...,::-1]# bgr to rgb
#     im[im!=im] = 0 # remove nan
#     im = im[...,:3] #只保留前3通道
    im = normalize_for_IGSNRR(im)
    if gt_path:
        gt = imread(gt_path)
    height, width, channel = im.shape
    nrow = np.ceil(height/imgcHeight).astype(np.int)
    ncol = np.ceil(width/imgcWidth).astype(np.int)
    row_overlap_length = int((nrow*imgcHeight-height)//(nrow-1))
    col_overlap_length = int((ncol*imgcWidth-width)//(ncol-1))
    row_list = [0]
    col_list = [0]
    for i in range(1,nrow-1):
        row_list.append(i*imgcHeight-row_overlap_length)
    row_list.append(height-imgcHeight)
    for i in range(1,ncol-1):
        col_list.append(i*imgcWidth-col_overlap_length)
    col_list.append(width-imgcWidth)
    all_left_point = list(itertools.product(row_list,col_list))
    cut_img_list = []
    cut_gt_list = []
    for point in all_left_point:
        x,y = point
        cut_img_list.append(np.copy(im)[x:x+imgcHeight, y:y+imgcWidth, :])
        if gt_path:
            cut_gt_list.append(np.copy(gt)[x:x+imgcHeight, y:y+imgcWidth].astype(np.uint8))
    if gt_path:
        return cut_img_list, cut_gt_list
    else:
        return cut_img_list

def normalize_gt(gt):
    for id in id_to_trainid:
        gt[gt==id] = id_to_trainid[id]
    gt = gt-1 # 将标签从1-17转换为0-16
    color_img = get_color_pallete(gt)
    return color_img

if __name__=='__main__':
    # cut for train set
    save_dir = '../data/IGSNRR/train'
    img_suffix = '.tif'
    gt_suffix = '.png'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in range(20):
        if i in [18,  2, 12, 10]:
            continue
        fp = '../data/labeled_data/task{:02d}/costal_S2_{}.tif'.format(i+1,i+1)
        cut_img_list, cut_gt_list = img_cut_spcific_num(fp, gt_path=fp.replace('S2', 'class'))
        for j in range(len(cut_img_list)):
            cur_file_name = '{:04d}_{:04d}'.format(i+1, j)
    #         print(cur_file_name)
            imsave(os.path.join(save_dir, cur_file_name+img_suffix), cut_img_list[j])
            imsave(os.path.join(save_dir, cur_file_name+gt_suffix), cut_gt_list[j])
    image_list = [i for i in os.listdir(save_dir) if i.endswith('.tif')]
    print('sucess save {} train images in {}'.format(len(image_list), save_dir))

    # cut for val set
    save_dir = '../data/IGSNRR/val'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    for i in [18,  2, 12, 10]:
        fp = '../data/labeled_data/task{:02d}/costal_S2_{}.tif'.format(i+1,i+1)
        cut_img_list, cut_gt_list = img_cut_spcific_num(fp, gt_path=fp.replace('S2', 'class'))
        for j in range(len(cut_img_list)):
            cur_file_name = '{:04d}_{:04d}'.format(i+1, j)
    #         print(cur_file_name)
            imsave(os.path.join(save_dir, cur_file_name+img_suffix), cut_img_list[j])
            imsave(os.path.join(save_dir, cur_file_name+gt_suffix), cut_gt_list[j])
    image_list = [i for i in os.listdir(save_dir) if i.endswith('.tif')]
    print('sucess save {} validation images in {}'.format(len(image_list), save_dir))