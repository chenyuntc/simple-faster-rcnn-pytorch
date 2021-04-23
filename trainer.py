from __future__ import  absolute_import
import os
from collections import namedtuple
import time
from torch.nn import functional as F
from model.utils.creator_tool import AnchorTargetCreator, ProposalTargetCreator

from torch import nn
import torch as t
from utils import array_tool as at
from utils.vis_tool import Visualizer

from utils.config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter

LossTuple = namedtuple('LossTuple',
                       ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss',
                        'total_loss'
                        ])

# trainer.py里面实现了很多函数供train.py调用
# __init__ Faster_RCNNTrainer的初始化函数，其父类是nn.module，主要是一些变量的初始化部分，定义了self.faster_rcnn = faster_rcnn,而这个rpn_sigma和roi_sigma是在_faster_rcnn_loc_loss调用用来计算位置损失函数用到的超参数，
# 之后定义了十分重要的两个函数，AnchorTargetCreator()和ProposalTargetCreator(),它们一个用于从20000个候选anchor中产生256个anchor进行二分类和位置回归，也就是为rpn网络产生的预测位置和预测类别提供真正的ground_truth标准
# 用于rpn网络的自我训练，自我提高，提升产生ROIs的精度！具体的筛选过程和准则看前面几篇文章，而ProposalTargetCreator()的作用是从2000个筛选出的ROIS中再次选出128个ROIs用于训练，它的作用和前面的anchortargetCreator类似，
# 不过它们服务的网络是不同的，前面anchortargetCreator服务的是RPN网络，而我们的proposaltargetCreator服务的是ROIHearder的网络，ROIheader的作用就是真正产生ROI__loc和ROI_cls的网络，它完成了目标检测最重要的预测目标位置和类别！
# 之后定义了位置信息的均值方差，因为送入到网络训练的位置信息全部是归一化处理的，需要用到相关的均值和方差数据，接下来是优化器数据，用的是faster_rcnn文件里的get_optimizer()数据，里面决定了是使用Adam还是SGD等等，以及衰减率的设置之类，
# 最后是可视化部分的一些设置，rpn_cm是混淆矩阵，就是验证预测值与真实值精确度的矩阵ConfusionMeter(2)括号里的参数指的是类别数，所以rpn_cm =2,而roi_cm =21因为roi的类别有21种（20个object类+1个background）
class FasterRCNNTrainer(nn.Module):
    """wrapper for conveniently training. return losses

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.
    * :obj:`total_loss`: The sum of 4 loss above.

    Args:
        faster_rcnn (model.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
    """

    def __init__(self, faster_rcnn):
        super(FasterRCNNTrainer, self).__init__()

        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        # target creator create gt_bbox gt_label etc as training targets. 
        self.anchor_target_creator = AnchorTargetCreator()
        self.proposal_target_creator = ProposalTargetCreator()

        self.loc_normalize_mean = faster_rcnn.loc_normalize_mean
        self.loc_normalize_std = faster_rcnn.loc_normalize_std

        self.optimizer = self.faster_rcnn.get_optimizer()
        # visdom wrapper
        self.vis = Visualizer(env=opt.env)

        # indicators for training status
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k: AverageValueMeter() for k in LossTuple._fields}  # average loss

    # pic6.png
    # 整幅图片描述在求损失之前训练过程经历了什么！不准确的说是一个伪正向传播的过程，为啥说是伪正向传播呢，因为过程中调用了proposal_target_creator()，
    # 而这个函数的作用其实是为了训练ROI_Header网络而提供所谓的128张sample_roi以及它的ground_truth的位置和label用的！所以它的根本目的是为了训练网络，在测试的时候是用不到的！
    # 流程图中红色圆框代表的是网络运行过程中产生的参数，而蓝色框代表的是网络定义的时候就有的参数！仔细看整个流程图，网络的运作结构就一目了然了！下面解释下代码：
    # n= bboxes.shape[0]首先获取batch个数，如果不等于就报错，因为本程序只支持batch_size=1,接着读取图片的高和宽，这里解释下，不论图片还是bbox，它们的数据格式都是形如n,c,hh,ww这种，所以H，W就可以获取到图片的尺寸，
    # 紧接着用self.faster_rcnn.extractor(imgs)提取图片的特征，然后放到rpn网络里面self.faster_rcnn.rpn(feature,img_size,scale)提取出rpn_locs,rpn_scores,rois,roi_indices,anchor来，
    # 下一步就是经过proposal_target_creator网络产生采样过后的sample_roi,以及其对应的gt_cls_loc和gt_score，最后经过head网络，完成整个的预测过程！流程图中的结构是一模一样的！
    # 但是这个文件之所以叫trainer就是因为不仅仅有正向的运作过程，肯定还有反向的传播，包括了损失计算等等，没错，接下来我们看下面的损失计算部分的流程图
    # pic7.png
    # 如上图所示，其实剩下的代码就是计算了两部分的损失，一个是RPN_losses,一个是ROI_Losses，为啥要这样做呢？大家考虑一下，这个Faster-rcnn的网络，哪些地方应用到了网络呢？一个是提取proposal的过程，
    # 在faster-rcnn里创造性的提出了anchor，用网络来产生proposals，所以rpn_losses就是为了计算这部分的损失，从而使用梯度下降的办法来提升提取prososal的网络的性能，另一个使用到网络的地方就是ROI_header，
    # 没错就是在利用特征图和ROIs来预测目标检测的类别以及位置的偏移量的时候再一次使用到了网络，那这部分预测网络的性能如何保证呢？ROI_losses就是计算这部分的损失函数，从而用梯度下降的办法来继续提升网络的性能
    # 这样一来，这两部分的网络的损失都记算出来了！forward函数也就介绍完了！这个地方需要特别注意的一点就是rpn_cm和roi_cm这两个对象应该是Confusion matrix也就是混淆矩阵啦，作用就是用于后续的数据可视化
    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~torch.autograd.Variable): A variable with a batch of images.
            bboxes (~torch.autograd.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~torch.autograd..Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            namedtuple of 5 losses
        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)

        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)

        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        # Sample RoIs and forward
        # it's fine to break the computation graph of rois, 
        # consider them as constant input
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
            roi,
            at.tonumpy(bbox),
            at.tonumpy(label),
            self.loc_normalize_mean,
            self.loc_normalize_std)
        # NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
            features,
            sample_roi,
            sample_roi_index)

        # ------------------ RPN losses -------------------#
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
            at.tonumpy(bbox),
            anchor,
            img_size)
        gt_rpn_label = at.totensor(gt_rpn_label).long()
        gt_rpn_loc = at.totensor(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
            rpn_loc,
            gt_rpn_loc,
            gt_rpn_label.data,
            self.rpn_sigma)

        # NOTE: default value of ignore_index is -100 ...
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(), ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label > -1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label) > -1]
        self.rpn_cm.add(at.totensor(_rpn_score, False), _gt_rpn_label.data.long())

        # ------------------ ROI losses (fast rcnn loss) -------------------#
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0, n_sample).long().cuda(), \
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.totensor(gt_roi_label).long()
        gt_roi_loc = at.totensor(gt_roi_loc)

        roi_loc_loss = _fast_rcnn_loc_loss(
            roi_loc.contiguous(),
            gt_roi_loc,
            gt_roi_label.data,
            self.roi_sigma)

        roi_cls_loss = nn.CrossEntropyLoss()(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score, False), gt_roi_label.data.long())

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]

        return LossTuple(*losses)

    # 整个函数实际上就是进行了一次参数的优化过程，首先self.optimizer.zero_grad()将梯度数据全部清零，然后利用刚刚介绍的self.forward(imgs,bboxes,labels,scales)函数将所有的损失计算出来，
    # 接着进行依次losses.total_loss.backward()反向传播计算梯度，self.optimizer.step()进行一次参数更新过程，self.update_meters(losses)就是将所有损失的数据更新到可视化界面上,最后将losses返回！
    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses = self.forward(imgs, bboxes, labels, scale)
        losses.total_loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses

    def save(self, save_optimizer=False, save_path=None, **kwargs):
        """serialize models include optimizer and other info
        return path where the model-file is stored.

        Args:
            save_optimizer (bool): whether save optimizer.state_dict().
            save_path (string): where to save model, if it's None, save_path
                is generate using time str and info from kwargs.
        
        Returns:
            save_path(str): the path to save models.
        """
        save_dict = dict()

        save_dict['model'] = self.faster_rcnn.state_dict()
        save_dict['config'] = opt._state_dict()
        save_dict['other_info'] = kwargs
        save_dict['vis_info'] = self.vis.state_dict()

        if save_optimizer:
            save_dict['optimizer'] = self.optimizer.state_dict()

        if save_path is None:
            timestr = time.strftime('%m%d%H%M')
            save_path = 'checkpoints/fasterrcnn_%s' % timestr
            for k_, v_ in kwargs.items():
                save_path += '_%s' % v_

        save_dir = os.path.dirname(save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        t.save(save_dict, save_path)
        self.vis.save([self.vis.env])
        return save_path

    def load(self, path, load_optimizer=True, parse_opt=False, ):
        state_dict = t.load(path)
        if 'model' in state_dict:
            self.faster_rcnn.load_state_dict(state_dict['model'])
        else:  # legacy way, for backward compatibility
            self.faster_rcnn.load_state_dict(state_dict)
            return self
        if parse_opt:
            opt._parse(state_dict['config'])
        if 'optimizer' in state_dict and load_optimizer:
            self.optimizer.load_state_dict(state_dict['optimizer'])
        return self

    def update_meters(self, losses):
        loss_d = {k: at.scalar(v) for k, v in losses._asdict().items()}
        for key, meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key, meter in self.meters.items():
            meter.reset()
        self.roi_cm.reset()
        self.rpn_cm.reset()

    def get_meter_data(self):
        return {k: v.value()[0] for k, v in self.meters.items()}

#  这个函数其实就是写了一个smooth_l1损失函数的计算公式，这个公式里面的x,t就是代表预测和实际的两个变量，
#  in_weight代表的是权重，因为在计算损失函数的过程中被标定为背景的那一类其实是不计算损失函数的，所以说可以巧妙地将对应的权重设置为0,
#  这样就完成了忽略背景类的目的，这也就是为什么计算位置的损失函数还要传入ground_truth的label作为参数的原因，sigma是一个因子，在前面的__init__函数里有定义好！
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

#  这个函数完成的任务就是用in_weight来作为权重，只将那些不是背景的anchor/ROIs的位置加入到损失函数的计算中来，
#  方法就是只给不是背景的anchor/ROIs的in_weight设置为1,这样就可以完成loc_loss的求和计算，最后进行返回就完成了计算位置损失的任务！
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    # NOTE:  unlike origin implementation, 
    # we don't need inside_weight and outside_weight, they can calculate by gt_label
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight).cuda()] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= ((gt_label >= 0).sum().float()) # ignore gt_label==-1 for rpn_loss
    return loc_loss
