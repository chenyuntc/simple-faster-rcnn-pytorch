import numpy as np

import chainer
import chainer.functions as F
import chainer.links as L

from chainercv.links.model.faster_rcnn.faster_rcnn import FasterRCNN
from chainercv.links.model.faster_rcnn.region_proposal_network import \
    RegionProposalNetwork
from chainercv.links.model.vgg.vgg16 import VGG16
from chainercv.utils import download_model

from torch import nn
from torchvision.models import vgg16


class FasterRCNNVGG16(FasterRCNN):

    """Faster R-CNN based on VGG-16.

    When you specify the path of a pre-trained chainer model serialized as
    a :obj:`.npz` file in the constructor, this chain model automatically
    initializes all the parameters with it.
    When a string in prespecified set is provided, a pretrained model is
    loaded from weights distributed on the Internet.
    The list of pretrained models supported are as follows:

    * :obj:`voc07`: Loads weights trained with the trainval split of \
        PASCAL VOC2007 Detection Dataset.
    * :obj:`imagenet`: Loads weights trained with ImageNet Classfication \
        task for the feature extractor and the head modules. \
        Weights that do not have a corresponding layer in VGG-16 \
        will be randomly initialized.

    For descriptions on the interface of this model, please refer to
    :class:`~chainercv.links.model.faster_rcnn.FasterRCNN`.

    :class:`~chainercv.links.model.faster_rcnn.FasterRCNNVGG16`
    supports finer control on random initializations of weights by arguments
    :obj:`vgg_initialW`, :obj:`rpn_initialW`, :obj:`loc_initialW` and
    :obj:`score_initialW`.
    It accepts a callable that takes an array and edits its values.
    If :obj:`None` is passed as an initializer, the default initializer is
    used.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        pretrained_model (str): The destination of the pre-trained
            chainer model serialized as a :obj:`.npz` file.
            If this is one of the strings described
            above, it automatically loads weights stored under a directory
            :obj:`$CHAINER_DATASET_ROOT/pfnet/chainercv/models/`,
            where :obj:`$CHAINER_DATASET_ROOT` is set as
            :obj:`$HOME/.chainer/dataset` unless you specify another value
            by modifying the environment variable.
        min_size (int): A preprocessing paramter for :meth:`prepare`.
        max_size (int): A preprocessing paramter for :meth:`prepare`.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.
        vgg_initialW (callable): Initializer for the layers corresponding to
            the VGG-16 layers.
        rpn_initialW (callable): Initializer for Region Proposal Network
            layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.
        proposal_creator_params (dict): Key valued paramters for
            :class:`~chainercv.links.model.faster_rcnn.ProposalCreator`.

    """

    _models = {
        'voc07': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.4/'
            'faster_rcnn_vgg16_voc07_trained_2017_08_06.npz'
        },
        'voc0712': {
            'n_fg_class': 20,
            'url': 'https://github.com/yuyu2172/share-weights/releases/'
            'download/0.0.4/faster_rcnn_vgg16_voc0712_trained_2017_07_21.npz'
        },
    }
    feat_stride = 16

    def __init__(self,
                 n_fg_class=None,
                 pretrained_model=None,
                 min_size=600, max_size=1000,
                 ratios=[0.5, 1, 2], anchor_scales=[8, 16, 32],
                 vgg_initialW=None, rpn_initialW=None,
                 loc_initialW=None, score_initialW=None,
                 proposal_creator_params=dict()
                 ):
        if n_fg_class is None:
            if pretrained_model not in self._models:
                raise ValueError(
                    'The n_fg_class needs to be supplied as an argument')
            n_fg_class = self._models[pretrained_model]['n_fg_class']

        extractor,classifier = decom_vgg16()

        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
            initialW=rpn_initialW,
            proposal_creator_params=proposal_creator_params,
        )

        head = VGG16RoIHead(
            n_fg_class + 1,
            roi_size=7, spatial_scale=1. / self.feat_stride,
            vgg_initialW=vgg_initialW,
            loc_initialW=loc_initialW,
            score_initialW=score_initialW
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
            mean=np.array([122.7717, 115.9465, 102.9801],
                          dtype=np.float32)[:, None, None],
            min_size=min_size,
            max_size=max_size
        )


def decom_vgg16(pretrained=True):
    # the 30th layer of features is relu of conv5_3
    model = vgg16(pretrained)
    features = list(model.features)[:30]
    classifier = model.classifier
    del classifier._modules['6']
    return nn.Sequential(features),classifier

class VGG16RoIHead(nn.Module):

    """Faster R-CNN Head for VGG-16 based implementation.

    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.

    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        vgg_initialW (callable): Initializer for the layers corresponding to
            the VGG-16 layers.
        loc_initialW (callable): Initializer for the localization head.
        score_initialW (callable): Initializer for the score head.

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 vgg_initialW=None, loc_initialW=None, score_initialW=None):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        #NOTE： 这里初始化的方式都被我修改，使用默认的初始化方式

        self.fc6 = nn.Linear(25088, 4096)
        self.fc7 = nn.Linear(4096, 4096, initialW = vgg_initialW)
        self.cls_loc = nn.Linear(4096, n_class * 4, initialW=loc_initialW)
        self.score = nn.Linear(4096, n_class, initialW=score_initialW)

        self.n_class = n_class
        self.roi_size = roi_size
        self.spatial_scale = spatial_scale

    def __call__(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        roi_indices = roi_indices.float()
        indices_and_rois = t.cat(roi_indices[:,None],rois,dim=1)

        ### TODO: implemented roi_pooling
        pool = _roi_pooling_2d_yx(
            x, indices_and_rois, self.roi_size, self.roi_size,
            self.spatial_scale)

        fc6 = F.relu(self.fc6(pool))
        fc7 = F.relu(self.fc7(fc6))
        roi_cls_locs = self.cls_loc(fc7)
        roi_scores = self.score(fc7)
        return roi_cls_locs, roi_scores


def _roi_pooling_2d_yx(x, indices_and_rois, outh, outw, spatial_scale):
    xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
    pool = F.roi_pooling_2d(
        x, xy_indices_and_rois, outh, outw, spatial_scale)
    return pool
