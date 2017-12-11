from collections import namedtuple
import numpy as np

from torch.nn import functional as F
from model.utils.target_tool import AnchorTargetCreator,ProposalTargetCreator

from torch import nn
import torch as t
from torch.autograd import Variable
from util.visulizer import Visualizer
from util import array_tool as at
from util.vis_tool import visdom_bbox

import matplotlib 
matplotlib.use('agg')

from config import opt
from torchnet.meter import ConfusionMeter, AverageValueMeter


LossTuple = namedtuple('LossTuple',
                        ['rpn_loc_loss',
                        'rpn_cls_loss',
                        'roi_loc_loss',
                        'roi_cls_loss'])

class FasterRCNNTrainer(nn.Module):
    
    """为训练FasterRCNN而做的封装，返回loss
    Calculate losses for Faster R-CNN and report them.
     
    This is used to train Faster R-CNN in the joint training scheme
    [#FRCNN]_.

    The losses include:

    * :obj:`rpn_loc_loss`: The localization loss for \
        Region Proposal Network (RPN).
    * :obj:`rpn_cls_loss`: The classification loss for RPN.
    * :obj:`roi_loc_loss`: The localization loss for the head module.
    * :obj:`roi_cls_loss`: The classification loss for the head module.

    .. [#FRCNN] Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun. \
    Faster R-CNN: Towards Real-Time Object Detection with \
    Region Proposal Networks. NIPS 2015.

    Args:
        faster_rcnn (~chainercv.links.model.faster_rcnn.FasterRCNN):
            A Faster R-CNN model that is going to be trained.
        rpn_sigma (float): Sigma parameter for the localization loss
            of Region Proposal Network (RPN). The default value is 3,
            which is the value used in [#FRCNN]_.
        roi_sigma (float): Sigma paramter for the localization loss of
            the head. The default value is 1, which is the value used
            in [#FRCNN]_.
        anchor_target_creator: An instantiation of
            :class:`~chainercv.links.model.faster_rcnn.AnchorTargetCreator`.
        proposal_target_creator_params: An instantiation of
            :class:`~chainercv.links.model.faster_rcnn.ProposalTargetCreator`.
    """

    def __init__(self, faster_rcnn, 
                 anchor_target_creator=AnchorTargetCreator(),
                 proposal_target_creator=ProposalTargetCreator(),
                 optimizer=None
                 ):
        super(FasterRCNNTrainer, self).__init__()
        self.opt = opt
        
        self.faster_rcnn = faster_rcnn
        self.rpn_sigma = opt.rpn_sigma
        self.roi_sigma = opt.roi_sigma

        self.anchor_target_creator = anchor_target_creator
        self.proposal_target_creator = proposal_target_creator

        self.loc_normalize_mean = opt.loc_normalize_mean
        self.loc_normalize_std = opt.loc_normalize_std
        
        self.optimizer = optimizer
        if self.optimizer is None:
            self.optimizer = self.faster_rcnn.get_optimizer()
            # t.optim.SGD(self.parameters(),lr = opt.lr,
            #                             momentum=0.9,
            #                             weight_decay=opt.weight_decay)
        
        self.vis = Visualizer(env=opt.env)
        self.rpn_cm = ConfusionMeter(2)
        self.roi_cm = ConfusionMeter(21)
        self.meters = {k:AverageValueMeter() for k in LossTuple._fields} # average loss


    def forward(self, imgs, bboxes, labels, scale):
        """Forward Faster R-CNN and calculate losses.

        Here are notations used.

        * :math:`N` is the batch size.
        * :math:`R` is the number of bounding boxes per image.

        Currently, only :math:`N=1` is supported.

        Args:
            imgs (~chainer.Variable): A variable with a batch of images.
            bboxes (~chainer.Variable): A batch of bounding boxes.
                Its shape is :math:`(N, R, 4)`.
            labels (~chainer.Variable): A batch of labels.
                Its shape is :math:`(N, R)`. The background is excluded from
                the definition, which means that the range of the value
                is :math:`[0, L - 1]`. :math:`L` is the number of foreground
                classes.
            scale (float or ~chainer.Variable): Amount of scaling applied to
                the raw image during preprocessing.

        Returns:
            chainer.Variable:
            Scalar loss variable.
            This is the sum of losses for Region Proposal Network and
            the head module.

        """
        n = bboxes.shape[0]
        if n != 1:
            raise ValueError('Currently only batch size 1 is supported.')

        _, _, H, W = imgs.shape
        img_size = (H, W)

        features = self.faster_rcnn.extractor(imgs)
        if self.faster_rcnn.lr1==0:features.detach() # detach to speed
        rpn_locs, rpn_scores, rois, roi_indices, anchor = \
            self.faster_rcnn.rpn(features, img_size, scale)
        # self.rpn_locs,self.rpn_score,self.rois,self.roi_indices,self.anchor = \
        #     at.totensor(rpn_locs), at.totensor(rpn_scores), at.totensor(rois), at.totensor(roi_indices), at.totensor(anchor)
        # Since batch size is one, convert variables to singular form
        bbox = bboxes[0]
        label = labels[0]
        rpn_score = rpn_scores[0]
        rpn_loc = rpn_locs[0]
        roi = rois

        

        # I think it's fine to break the computation graph of rois
        # Sample RoIs and forward
        sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(
                                roi, 
                                at.tonumpy(bbox),
                                at.tonumpy(label),
                                self.loc_normalize_mean, 
                                self.loc_normalize_std)

        #NOTE it's all zero because now it only support for batch=1 now
        sample_roi_index = t.zeros(len(sample_roi))
        roi_cls_loc, roi_score = self.faster_rcnn.head(
                                        features, 
                                        sample_roi, 
                                        sample_roi_index)
        
        # RPN losses
        gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(
                                            at.tonumpy(bbox), 
                                            anchor, 
                                            img_size)
        gt_rpn_label = at.tovariable(gt_rpn_label).long()
        gt_rpn_loc = at.tovariable(gt_rpn_loc)
        rpn_loc_loss = _fast_rcnn_loc_loss(
                            rpn_loc, 
                            gt_rpn_loc, 
                            gt_rpn_label.data, 
                            self.rpn_sigma)
            
        # NOTE: why default ignore_index is -100.......
        rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_label.cuda(),ignore_index=-1)
        _gt_rpn_label = gt_rpn_label[gt_rpn_label>-1]
        _rpn_score = at.tonumpy(rpn_score)[at.tonumpy(gt_rpn_label)>-1]
        self.rpn_cm.add(at.totensor(_rpn_score,False), _gt_rpn_label.data.long())

        # Losses for outputs of the head.
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, -1, 4)
        roi_loc = roi_cls_loc[t.arange(0,n_sample).long().cuda(),\
                              at.totensor(gt_roi_label).long()]
        gt_roi_label = at.tovariable(gt_roi_label).long()
        gt_roi_loc = at.tovariable(gt_roi_loc)

        # roi_weight = 1./ (self.roi_cm.value().sum(1)+1)
        # roi_weight  = F.softmax(roi_weight)#.sum()
        # self.roi_loss = nn.CrossEntropyLoss(at.totensor(roi_weight).float())

        roi_loc_loss = _fast_rcnn_loc_loss(
                            roi_loc, 
                            gt_roi_loc, 
                            gt_roi_label.data,
                            self.roi_sigma)
        self.roi_loss = nn.CrossEntropyLoss()
        # gt_roi_label = Variable(gt_roi_label)
        roi_cls_loss =self.roi_loss(roi_score, gt_roi_label.cuda())

        self.roi_cm.add(at.totensor(roi_score,False), gt_roi_label.data.long())

        losses = rpn_loc_loss , rpn_cls_loss , roi_loc_loss , roi_cls_loss

        return LossTuple(*losses),rois

    def train_step(self, imgs, bboxes, labels, scale):
        self.optimizer.zero_grad()
        losses,rois = self.forward(imgs, bboxes, labels, scale)
        rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss = losses
        loss = sum(losses)
        loss.backward()
        self.optimizer.step()
        self.update_meters(losses)
        return losses,rois

    # def visulize(self):


    def update_meters(self, losses):
        loss_d = {k:at.scalar(v) for k,v in losses._asdict().items()}
        for key,meter in self.meters.items():
            meter.add(loss_d[key])

    def reset_meters(self):
        for key,meter in self.meters.items():
            meter.reset()

    def get_meter_data(self):
        return {k:v.value()[0] for k,v in self.meters.items()}

        

def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = (diff).abs()
    flag = (abs_diff.data < (1. / sigma2)).float()
    flag = Variable(flag)
    y = (flag * (sigma2 / 2.) * (diff**2) +
         (1 - flag) * (abs_diff - 0.5 / sigma2))

    return (y).sum()


def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):

    in_weight = t.zeros(gt_loc.shape).cuda()
    # Localization loss is calculated only for positive rois.
    #### NOTE:  something is wrong or somthing is different to origian implementation
    in_weight[(gt_label > 0).view(-1,1).expand_as(in_weight).cuda()] = 1
    #####
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, Variable(in_weight), sigma)
    # Normalize by total number of negtive and positive rois.
    loc_loss /= (gt_label >= 0).sum()
    return loc_loss

