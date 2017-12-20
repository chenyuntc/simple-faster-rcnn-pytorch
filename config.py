from pprint import pprint


# Default Configs for training
# NOTE that, config items could be overwriten by passing argument through command line.
# e.g. --voc-data-dir='./data/'

class Config:
    # data
    voc_data_dir = '/mnt/3/VOC/VOCdevkit/VOC2007/'
    voc_data_dir = '/home/cy/.chainer/dataset/pfnet/chainercv/voc/VOCdevkit/VOC2007/'
    min_size = 600
    max_size = 1000
    num_workers = 8

    # sigma for l1_smooth_loss
    rpn_sigma = 3.
    roi_sigma = 1.

    # param for optimizer
    weight_decay = 0.0005  # 0.0005 in origin paper but 0.0001 in tf-faster-rcnn
    lr_decay = 0.1  # 1e-3 -> 1e-4
    # lr = 1e-3
    lr1 = 1e-3  # extractor
    lr2 = 1e-3  # rpn
    lr3 = 1e-3  # roi head

    # visualization
    env = 'faster-rcnn'  # visdom env
    port = 8097
    plot_every = 40  # vis every N iter

    # preset
    data = 'voc'
    pretrained_model = 'vgg16'

    # training
    epoch = 100

    # change lr
    milestone = [0, 1, 5, 10]

    use_adam = False

    # debug
    debug_file = '/tmp/debugf'

    test_num = 10000
    # model
    load_path = None  # '/mnt/3/rpn.pth'

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
