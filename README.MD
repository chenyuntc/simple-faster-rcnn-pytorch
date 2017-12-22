# A Pythonic, Extensible and Minimal Implemention of Faster RCNN Without Harming Performance

##  Introduction

This project is a **Simplified** Faster RCNN implementation based on [chainercv](https://github.com/chainer/chainercv) and other [projects](#Acknowledgement) . It aims to:

- Simplify the code (*Simple is better than complex*)
- Make the code more straight forward (*Flat is better than nested*)
- Match the performance reported in [ origin paper](https://arxiv.org/abs/1506.01497) (*Speed Counts and mAP Matters*)

##  Performance

- mAP

VGG16 train on `trainval` and test on `test` split. 

**Note**: the training shows great randomness, you may need to train more epoch to reach the highest mAP. However, it should be easy to reach the lowerboud. 

|              Implementation              |     mAP     |
| :--------------------------------------: | :---------: |
| [origin paper](https://arxiv.org/abs/1506.01497) |    0.699    |
| using caffe pretrained model (enable with`--caffe-pretrain`) | 0.702-0.712 |
|    using torchvision pretrained model    | 0.693-0.701 |
| model converted from [chainercv](https://github.com/chainer/chainercv/tree/master/examples/faster_rcnn) (reported 0.706) |   0.7053    |


- Speed

|              Implementation              |   GPU    | Inference | Trainining |
| :--------------------------------------: | :------: | :-------: | :--------: |
| [origin paper](https://arxiv.org/abs/1506.01497) |   K40    |   5 fps   |     NA     |
|                   This                   | TITAN Xp | 12 fps^*^ |  5-6 fps   |
| [pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) | TITAN Xp |    NA     | 5-6fps^**^ |

\* include reading images from disk, preprocessing, etc. see `eval` in `train.py` for more detail.

** it depends on the environment.

**NOTE ** you should make sure you install cupy correctly to reach the benchmark.

## Install denpendencies

- install PyTorch >=0.3 with GPU (code are gpu-only), refer to [official website](http://pytorch.org)
- install cupy, you can install via `pip install` but it's better to read the [docs](https://docs-cupy.chainer.org/en/latest/install.html#install-cupy-with-cudnn-and-nccl) and make sure the environ is correctly set


- install other dependencies:  `pip install -r requirements.txt `
- build `nms_gpu_post`: `cd model/utils/nms/; python3 build.py build_ext --inplace`
- start vidom for visualize


```
nohup python3 -m visdom.server &
```

If you're in China and have encounter problem with visdom (i.e. timeout, blank screen), you may refer to [visdom issue](https://github.com/facebookresearch/visdom/issues/111#issuecomment-321743890), ~~and a temporay solution provided by me~~

## Demo

Download pretrained model from [google drive](https://drive.google.com/open?id=1cQ27LIn-Rig4-Uayzy_gH5-cW-NRGVzY).


See [demo.ipynb](https://github.com/chenyuntc/Simplified-FasterRCNN/blob/master/demo.ipynb) for more detail.

## Train

### Data

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

### preprare caffe-pretrained vgg16

If you want to use caffe-pretrain model as initial weight, you can run below to get vgg16 weights converte from caffe, which is the same as the origin paper use.

````
python misc/convert_caffe_pretrain.py
````

Then you should speicified where caffe-pretraind model `vgg16_caffe.pth` stored in `config.py` by setting `caffe_pretrain_path`

If you want to use torchvision pretrained model, you may skip this step.

**NOTE that**, caffe pretrained model has showed slight better performance.

### begin traininig

```Bash
make checkpoints/ # make dir for storing snapshots
```

```bash
python3 train.py train --env='fasterrcnn-caffe' --plot-every=100 --caffe-pretrain	
```

you may refer to `config.py` for more argument.

Some Key arguments:

- `--caffe-pretrain`=True: use caffe pretrain model  or use torchvision pretrained model (Default: torchvison)
- `--plot-every=n`: visulize predict, loss etc every n batches.
- `--env`: visdom env for visulization
- `--voc_data_dir`: where the VOC data stored
- `--use-drop`: use dropout in roi head, default without dropout
- `--use-adam`: use adam instead of SGD, default SGD. (You need set a very low `lr` for Adam)
- `--load-path`: pretrained model path, default `None`, if it's specified, the pretrained model would be loaded.

## Troubleshooting

TODO: make it clear

- visdom
- dataloader/ulimit
- cupy
- vgg

## TODO
- [] training on coco
- [] resnet
- [] replace cupy with THTensor+cffi?

## Acknowledge
This work builds on many excellent works, which include:

- [Yusuke Niitani's ChainerCV](https://github.com/chainer/chainercv) (mainly)
- [Ruotian Luo's pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn) which based on [ Xinlei Chen's tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn)
- [faster-rcnn.pytorch by Jianwei Yang and Jiasen Lu](https://github.com/jwyang/faster-rcnn.pytorch).It's mainly based on [longcw's faster_rcnn_pytorch](https://github.com/longcw/faster_rcnn_pytorch)
- All the above Repositories  have refer to [py-faster-rcnn by Ross Girshick and Sean Bell](https://github.com/rbgirshick/py-faster-rcnn)  either directly or indirectly. 

## LICENSE

MIT, see the LICENSE for more detail.