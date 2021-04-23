from __future__ import  absolute_import
from __future__ import  division
import torch as t
from data.voc_dataset import VOCBboxDataset
from skimage import transform as sktsf
from torchvision import transforms as tvtsf
from data import util
import numpy as np
from utils.config import opt

# 函数首先读取opt.caffe_pretrain判断是否使用caffe_pretrain进行预训练如果是的话，
# 对图片进行逆正则化处理，就是将图片处理成caffe模型需要的格式
def inverse_normalize(img):
    if opt.caffe_pretrain:
        img = img + (np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1))
        return img[::-1, :, :]
    # approximate un-normalize for visualize
    return (img * 0.225 + 0.45).clip(min=0, max=1) * 255

# 函数首先设置归一化参数normalize=tvtsf.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
# 然后对图片进行归一化处理img=normalize(t.from_numpy(img))
def pytorch_normalze(img):
    """
    https://github.com/pytorch/vision/issues/223
    return appr -1~1 RGB
    """
    normalize = tvtsf.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    img = normalize(t.from_numpy(img))
    return img.numpy()

# caffe的图片格式是BGR，所以需要img[[2,1,0],:,:]将RGB转换成BGR的格式，然后图片img = img*255 ,
# mean = np.array([122.7717,115.9465,102.9801]).reshape(3,1,1)设置图片均值
# 然后用图片减去均值完成caffe形式的归一化处理
def caffe_normalize(img):
    """
    return appr -125-125 BGR
    """
    img = img[[2, 1, 0], :, :]  # RGB-BGR
    img = img * 255
    mean = np.array([122.7717, 115.9465, 102.9801]).reshape(3, 1, 1)
    img = (img - mean).astype(np.float32, copy=True)
    return img

# 图片处理函数，C,H,W = img.shape 读取图片格式通道，高度，宽度
# Scale1 = min_size/min(H,W)
# Scale2 = max_size / max(H,W)
# Scale = min(scale1,scale2)设置放缩比，这个过程很直觉，选小的方便大的和小的都能够放缩到合适的位置
# img  = img/ 255
# img = sktsf.resize(img,(C,H*scale,W*scale),model='reflecct')将图片调整到合适的大小位于(min_size,max_size)之间、
# 然后根据opt.caffe_pretrain是否存在选择调用前面的pytorch正则化还是caffe_pretrain正则化
def preprocess(img, min_size=600, max_size=1000):
    """Preprocess an image for feature extraction.

    The length of the shorter edge is scaled to :obj:`self.min_size`.
    After the scaling, if the length of the longer edge is longer than
    :param min_size:
    :obj:`self.max_size`, the image is scaled to fit the longer edge
    to :obj:`self.max_size`.

    After resizing the image, the image is subtracted by a mean image value
    :obj:`self.mean`.

    Args:
        img (~numpy.ndarray): An image. This is in CHW and RGB format.
            The range of its value is :math:`[0, 255]`.

    Returns:
        ~numpy.ndarray: A preprocessed image.

    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    scale = min(scale1, scale2)
    img = img / 255.
    img = sktsf.resize(img, (C, H * scale, W * scale), mode='reflect',anti_aliasing=False)
    # both the longer and shorter should be less than
    # max_size and min_size
    if opt.caffe_pretrain:
        normalize = caffe_normalize
    else:
        normalize = pytorch_normalze
    return normalize(img)

# 因为要对函数进行缩放或者扩大操作，是的短边<=600,长边<=1000,并且满足至少一边是等于
# __init__函数设置了图片的最小最大尺寸，本pytorch代码中min_size=600,max_size=1000
# __call__函数中 从in_data中读取 img,bbox,label 图片，bboxes的框框和label
# 然后从_,H,W = img.shape读取出图片的长和宽
# img = preposses(img,self.min_size,self.max_size)将图片进行最小最大化放缩然后进行归一化
# _,o_H,o_W = img.shape 读取放缩后图片的shape
# scale = o_H/H 放缩前后相除，得出放缩比因子
# bbox = util.reszie_bbox(bbox,(H,W),(o_H,o_W)) 重新调整bboxes框的大小
# img,params = utils.random_flip(img.x_random =True,return_param=True)进行图片的随机反转，图片旋转不变性，增强网络的鲁棒性！
# 同样的对bboxes进行随机反转，最后返回img,bbox,label,scale
class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, in_data):
        img, bbox, label = in_data
        _, H, W = img.shape
        img = preprocess(img, self.min_size, self.max_size)
        _, o_H, o_W = img.shape
        scale = o_H / H
        bbox = util.resize_bbox(bbox, (H, W), (o_H, o_W))

        # horizontally flip
        img, params = util.random_flip(
            img, x_random=True, return_param=True)
        bbox = util.flip_bbox(
            bbox, (o_H, o_W), x_flip=params['x_flip'])

        return img, bbox, label, scale


class Dataset:
    # opt.voc_data_dir = '/home/featurize/data/VOCdevkit/VOC2007/'
    #     min_size = 600  # image resize
    #     max_size = 1000 # image resize
    #     min_size 与 max_size 主要是因为我们图片要求 短边<=600 ,长边<= 1000
    def __init__(self, opt):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir)
        self.tsf = Transform(opt.min_size, opt.max_size)
    # —getitem__可以简单的理解为从数据集存储路径中将例子一个个的获取出来，
    # 然后调用前面的Transform函数将图片,label进行最小值最大值放缩归一化，
    # 重新调整bboxes的大小，然后随机反转，最后将数据集返回！
    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)

        img, bbox, label, scale = self.tsf((ori_img, bbox, label))
        # TODO: check whose stride is negative to fix this instead copy all
        # some of the strides of a given numpy array are negative.
        return img.copy(), bbox.copy(), label.copy(), scale

    def __len__(self):
        return len(self.db)

# TestData完成的功能和前面类似，但是获取调用的数据集是不同的，
# 因为def __init__(self,opt,split='test',use_difficult=True)
# 可以看到它在从Voc_data_dir中获取数据的时候使用了split='test'
# 也就是从test往后分割的部分数据送入到TestDataset的self.db中，
# 然后在进行图片处理的时候，并没有调用transform函数，
# 因为测试图片集没有bboxes需要考虑，同时测试图片集也不需要随机反转，
# 反转无疑为测试准确率设置了阻碍！所以直接调用preposses()函数进行最大值最小值裁剪然后归一化就完成了测试数据集的处理！
# 最后将整个self.db返回
class TestDataset:
    def __init__(self, opt, split='test', use_difficult=True):
        self.opt = opt
        self.db = VOCBboxDataset(opt.voc_data_dir, split=split, use_difficult=use_difficult)

    def __getitem__(self, idx):
        ori_img, bbox, label, difficult = self.db.get_example(idx)
        img = preprocess(ori_img)
        return img, ori_img.shape[1:], bbox, label, difficult

    def __len__(self):
        return len(self.db)
