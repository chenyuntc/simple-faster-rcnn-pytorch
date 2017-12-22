import os

import ipdb
import matplotlib
from tqdm import tqdm

import torch as t
from config import opt
from data.dataset import Dataset, TestDataset
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from util import array_tool as at
from util.vis_tool import visdom_bbox
from util.eval_tool import eval_detection_voc

matplotlib.use('agg')

def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0], sizes[1][0]]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict2(imgs, [sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii == test_num: break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=2,
                                       shuffle=False, \
                                       # pin_memory=True
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)

    # trainer.optimizer = trainer.faster_rcnn.get_great_optimizer()
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale, ori_img) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            img, bbox, label = Variable(img), Variable(bbox), Variable(label)
            losses = trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = (img * 0.225 + 0.45).clamp(min=0, max=1) * 255
                gt_img = visdom_bbox(at.tonumpy(ori_img_)[0], 
                                    at.tonumpy(bbox_)[0], 
                                    label_[0].numpy())
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(ori_img,visualize=True)
                pred_img = visdom_bbox( at.tonumpy(ori_img[0]), 
                                        at.tonumpy(_bboxes[0]),
                                        at.tonumpy(_labels[0]).reshape(-1), 
                                        at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        if best_map>0.6 and opt.test_num<5000: 
            opt.test_num=10000
            best_map = 0
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch==9:
            # trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
        # if epoch ==0:
        #     trainer.optimizer = trainer.faster_rcnn.get_optimizer()

        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                            str(eval_result['map']), 
                                            str(trainer.get_meter_data()))
        trainer.vis.log(log_info)
        # t.save(trainer.state_dict(),'checkpoints/fasterrcnn_%s.pth' %epoch)
        # t.vis.save([opt.env])


if __name__ == '__main__':
    import fire

    fire.Fire()
