# A Simple and Fast Implementation of Faster R-CNN

##  Introduction

This project is a **Simplified** Faster R-CNN implementation based on [chainercv](https://github.com/chainer/chainercv) and other [projects](#Acknowledgement) . It aims to:

- Simplify the code (*Simple is better than complex*)
- Make the code more straightforward (*Flat is better than nested*)
- Match the performance reported in [origin paper](https://arxiv.org/abs/1506.01497) (*Speed Counts and mAP Matters*)

##  Performance

### mAP

VGG16 train on `trainval` and test on `test` split. 

**Note**: the training shows great randomness, you may need a bit of luck and more epoches of training to reach the highest mAP. However, it should be easy to surpass the lower bound. 

|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
| [origin paper](https://arxiv.org/abs/1506.01497) |    0.699    |
|    train with caffe pretrained model     | 0.701-0.712 |
| train with torchvision pretrained model  | 0.685-0.701 |
| model converted from [chainercv](https://github.com/chainer/chainercv/tree/master/examples/faster_rcnn) (reported 0.706) |   0.7053    |

### Speed

|              Implementation              |   GPU    | Inference | Trainining |
| :--------------------------------------: | :------: | :-------: | :--------: |
| [origin paper](https://arxiv.org/abs/1506.01497) |   K40    |   5 fps   |     NA     |
|                 This[^1]                 | TITAN Xp |  12 fps   |  5-6 fps   |
| [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) | TITAN Xp |    NA     |   5-6fps   |

[^1]: make sure you install cupy correctly and only one program run on the GPU.
## Install dependencies

requires python3 and PyTorch 0.3

- install PyTorch >=0.3 with GPU (code are GPU-only), refer to [official website](http://pytorch.org)
- install cupy, you can install via `pip install` but it's better to read the [docs](https://docs-cupy.chainer.org/en/latest/install.html#install-cupy-with-cudnn-and-nccl) and make sure the environ is correctly set


- install other dependencies:  `pip install -r requirements.txt `
- build `nms_gpu_post`: `cd model/utils/nms/; python3 build.py build_ext --inplace`
- start vidom for visualize


```
nohup python3 -m visdom.server &
```

If you're in China and have encounter problem with visdom (i.e. timeout, blank screen), you may refer to [visdom issue](https://github.com/facebookresearch/visdom/issues/111#issuecomment-321743890), ~~and a temporary solution provided by me~~

## Demo

Download pretrained model from [Google Drive](https://drive.google.com/open?id=1cQ27LIn-Rig4-Uayzy_gH5-cW-NRGVzY).


See [demo.ipynb](https://github.com/chenyuntc/simple-faster-rcnn-pytorch/blob/master/demo.ipynb) for more detail.

## Train

### Prepare data

#### Pascal VOC2007

1. Download the training, validation, test data and VOCdevkit

   ```
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar
   wget http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar
   ```

2. Extract all of these tars into one directory named `VOCdevkit`

   ```
   tar xvf VOCtrainval_06-Nov-2007.tar
   tar xvf VOCtest_06-Nov-2007.tar
   tar xvf VOCdevkit_08-Jun-2007.tar
   ```

3. It should have this basic structure

   ```
   $VOCdevkit/                           # development kit
   $VOCdevkit/VOCcode/                   # VOC utility code
   $VOCdevkit/VOC2007                    # image sets, annotations, etc.
   # ... and several other directories ...
   ```

4.  specifiy the `voc_data_dir` in `config.py`, or pass it to program using argument like `--voc-data-dir=/path/to/VOCdevkit/VOC2007/` .

#### COCO

TBD

### Prepare caffe-pretrained vgg16

If you want to use caffe-pretrain model as initial weight, you can run below to get vgg16 weights converted from caffe, which is the same as the origin paper use.

````
python misc/convert_caffe_pretrain.py
````

This scripts would download pretrained model and converted it to the format compatible with torchvision. 

Then you should specify where caffe-pretraind model `vgg16_caffe.pth` stored in `config.py` by setting `caffe_pretrain_path`

If you want to use torchvision pretrained model, you may skip this step.

**NOTE**, caffe pretrained model has shown slight better performance.

**NOTE**: caffe model require images in BGR 0-255, while torchvision model requires images in RGB and 0-1. See `data/dataset.py`for more detail.

### begin training

```Bash
mkdir checkpoints/ # folder for snapshots
```

```bash
python3 train.py train --env='fasterrcnn-caffe' --plot-every=100 --caffe-pretrain    
```

you may refer to `config.py` for more argument.

Some Key arguments:

- `--caffe-pretrain`=True: use caffe pretrain model  or use torchvision pretrained model (Default: torchvison)
- `--plot-every=n`: visualize predict, loss etc every n batches.
- `--env`: visdom env for visualization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in ROI head, default without dropout
- `--use-Adam`: use Adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, the pretrained model would be loaded.


you may open browser, type:`http://<ip>:8097` and see the visualization of training procedure as below:
![visdom](http://7zh43r.com1.z0.glb.clouddn.com/del/visdom-fasterrcnn.png) 
## Troubleshooting

TODO: make it clear

- visdom
- dataloader/ulimit
- cupy
- vgg

## More

- [ ] training on coco
- [ ] resnet
- [ ] replace cupy with THTensor+cffi?
- [ ] Convert all numpy code to tensor?

## Acknowledgement
This work builds on many excellent works, which include:

- [Yusuke Niitani's ChainerCV](https://github.com/chainer/chainercv) (mainly)
- [Ruotian Luo's pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) which based on [ Xinlei Chen's tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
- [faster-rcnn.pytorch by Jianwei Yang and Jiasen Lu](https://github.com/jwyang/faster-rcnn.pytorch).It's mainly based on [longcw's faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)
- All the above Repositories have referred to [py-faster-rcnn by Ross Girshick and Sean Bell](https://github.com/rbgirshick/py-faster-rcnn)  either directly or indirectly. 

## other
licensed in MIT, see the LICENSE for more detail.
Contribution Welcome.
If you encounter any problem, feel free to open an issue.
Correct me if anything is wrong or unclear.