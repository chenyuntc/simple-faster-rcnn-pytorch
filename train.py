from __future__ import  absolute_import
import os

import ipdb
import matplotlib
from tqdm import tqdm

from utils.config import opt # opt 是 config引入的
from data.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

import resource

rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (20480, rlimit[1]))

matplotlib.use('agg')

#  eval()顾名思义，就是一个评估预测结果好坏的函数，展开来看果不其然，首先pred_bboxes,pred_labels,pred_scores ,gt_bboxes,gt_labels,gt_difficults 一开始就定义了这么多的list列表！它们分别是预测框的位置，预测框的类别和分数以及相应的真实值的类别分数等等！
# 接下来就是一个for循环，从 enumerate(dataloader)里面依次读取数据，读取的内容是: imgs图片，sizes尺寸，gt_boxes真实框的位置 gt_labels真实框的类别以及gt_difficults这些
# 然后利用faster_rcnn.predict(imgs,[sizes]) 得出预测的pred_boxes_,pred_labels_,pred_scores_预测框位置，预测框标记以及预测框的分数等等！这里的predict是真正的前向传播过程！完成真正的预测目的！
# 之后将pred_bbox,pred_label,pred_score ,gt_bbox,gt_label,gt_difficult预测和真实的值全部依次添加到开始定义好的列表里面去，如果迭代次数等于测试test_num，那么就跳出循环！调用 eval_detection_voc函数，
# 接收上述的六个列表参数，完成预测水平的评估！得到预测的结果！这个eval_detection_voc后面会解释
def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = list(), list(), list()
    gt_bboxes, gt_labels, gt_difficults = list(), list(), list()
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
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

# python train.py train --env='fasterrcnn' --plot_every=100
# env: fasterrcnn 主要是visdom显示的名字用
# plot-every: 100  经过 ${plot_every} 步后可视化一张图片
# 这俩参数都在 config.py中可以找到，plot_every 默认值是 40, 我们启动中改为 100, env传入值与默认值相同 都是 'fasterrcnn'
# 流程
# 1首先在可视化界面重设所有数据
# 2然后从训练数据中枚举dataloader,设置好缩放范围，将img,bbox,label,scale全部设置为可gpu加速
# 3调用trainer.py中的函数trainer.train_step(img,bbox,label,scale)进行一次参数迭代优化过程！
# 4 判断数据读取次数是否能够整除plot_every(是否达到了画图次数)，如果达到判断debug_file是否存在，用ipdb工具设置断点，调用trainer中的trainer.vis.plot_many(trainer.get_meter_data())将训练数据读取并上传完成可视化！
# 5将每次迭代读取的图片用dataset文件里面的inverse_normalize()函数进行预处理，将处理后的图片调用Visdom_bbox(ori_img_,at_tonumpy(_bboxes[0]),at.tonumpy(_labels[0].reshape(-1)),at.tonumpy(_scores[0]))
# 6调用trainer.vis.img('pred_img',pred_img)将迭代读取原始数据中的原图，bboxes框架，labels标签在可视化工具下显示出来
# 7调用 _bboxes,_labels,_socres = trainer.faster_rcnn.predict([ori_img_],visualize=True)调用faster_rcnn的predict函数进行预测，预测的结果保留在以_下划线开头的对象里面
# 8利用同样的方法将原始图片以及边框类别的预测结果同样在可视化工具中显示出来！
# 9调用train.vis.text(str(trainer.rpn_cm.value().tolist),win='rpn_cm')将rpn_cm也就是RPN网络的混淆矩阵在可视化工具中显示出来
# 10调用trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())将Roi_cm将roi的可视化矩阵以图片的形式显示出来
# ===============接下来是测试阶段的代码=============================================================
# 11 调用eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)将测试数据调用eval()函数进行评价，存储在eval_result中
# 12 trainer.vis.plot('test_map', eval_result['map']) 将eval_result['map']在可视化工具中进行显示
# 13  lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr'] 设置学习的learning rate
# 14log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),str(eval_result['map']),str(trainer.get_meter_data())) + trainer.vis.log(log_info) 将损失学习率以及map等信息及时显示更新
# 15 用if判断语句永远保存效果最好的map！
# 16 if判断语句如果学习的epoch达到了9就将学习率*0.1变成原来的十分之一
# 17 判断epoch==13结束训练验证过程
def train(**kwargs):
    # opt是config.py 引入的，config.py最后有 opt = config(),那么主要就是config的配置
    # kwargs 主要是 对注入 --XXX=XXX 格式变量进行解析 如--env='fasterrcnn' --plot_every=100，将其修改为新的值
    opt._parse(kwargs)

    # 可以看 Dataset函数的实现
    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset, \
                                  batch_size=1, \
                                  shuffle=True, \
                                  # pin_memory=True,
                                  num_workers=opt.num_workers)
    # 这里是去读测试集,其中 split默认为text表示读了text.txt
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                       batch_size=1,
                                       num_workers=opt.test_num_workers,
                                       shuffle=False, \
                                       pin_memory=False
                                       )
    faster_rcnn = FasterRCNNVGG16()
    print('model construct completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print('load pretrained model from %s' % opt.load_path)
    trainer.vis.text(dataset.db.label_names, win='labels')
    best_map = 0
    lr_ = opt.lr
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())

                # plot groud truth bboxes
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                     at.tonumpy(bbox_[0]),
                                     at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                       at.tonumpy(_bboxes[0]),
                                       at.tonumpy(_labels[0]).reshape(-1),
                                       at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = 'lr:{}, map:{},loss:{}'.format(str(lr_),
                                                  str(eval_result['map']),
                                                  str(trainer.get_meter_data()))
        trainer.vis.log(log_info)

        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
        if epoch == 9:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        if epoch == 13: 
            break


if __name__ == '__main__':
    import fire

    fire.Fire()
