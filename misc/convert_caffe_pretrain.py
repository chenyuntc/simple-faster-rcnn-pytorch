# code from ruotian luo
# https://github.com/ruotianluo/pytorch-faster-rcnn
import torch
from torch.utils.model_zoo import load_url
from torchvision import models

sd = load_url("https://s3-us-west-2.amazonaws.com/jcjohns-models/vgg16-00b39a1b.pth")
sd['classifier.0.weight'] = sd['classifier.1.weight']
sd['classifier.0.bias'] = sd['classifier.1.bias']
del sd['classifier.1.weight']
del sd['classifier.1.bias']

sd['classifier.3.weight'] = sd['classifier.4.weight']
sd['classifier.3.bias'] = sd['classifier.4.bias']
del sd['classifier.4.weight']
del sd['classifier.4.bias']

import  os
# speicify the path to save
if not os.path.exists('checkpoints'):
    os.makedirs('checkpoints')
torch.save(sd, "checkpoints/vgg16_caffe.pth")