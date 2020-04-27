import os
os.chdir('/content/drive/My Drive/lq_det_hyper/lq_det')


%reload_ext autoreload
%autoreload 2
import os
import torch as t
from utils.config import Config
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from data.util import  read_image
from utils.vis_tool import vis_bbox
from utils import array_tool as at
%matplotlib inline



img_name = 'demo.jpg'
raw_img = read_image(f'/content/drive/My Drive/lq_det_hyper/lq_det/misc/{img_name}')
raw_img = t.from_numpy(raw_img).unsqueeze(dim=0)



faster_rcnn = FasterRCNNVGG16()
trainer = FasterRCNNTrainer(faster_rcnn, using_visdom=False).cuda()


trainer.load('/content/drive/My Drive/lq_det_hyper/lq_det/ckpt/fasterrcnn_12222105_0.712649824453_caffe_pretrain.pth')
Config.caffe_vgg=True # this model was trained from caffe-pretrained model
_bboxes, _labels, _scores = trainer.faster_rcnn.predict(raw_img, visualize=True)
img, bbox, label, score = (at.tonumpy(raw_img[0]), at.tonumpy(_bboxes[0]), at.tonumpy(_labels[0]).reshape(-1), at.tonumpy(_scores[0]).reshape(-1))
vis_bbox(img, bbox, label, score)


import matplotlib.pyplot as plt
plt.show()