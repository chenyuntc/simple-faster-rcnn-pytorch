import numpy as np
import numpy as xp

import six
from six import __init__

#bbox_tools.py部分的代码主要由四个函数构成：1loc2bbox(src_bbox,loc)和bbox2loc(src_bbox,dst_bbox)是一对函数，其功能是刚好相反的，
# 比如loc2bbox()看其函数的参数src_bbox,loc就知道是有已知源框框和位置偏差，求出目标框框的作用，
# 而bbox2loc(src_bbox,dst_bbox)函数看其参数就知道是完成已知源框框和参考框框求出其位置偏差的功能！
# 而这个bbox_iou看函数名字我们也大概能猜出是求两个bbox的相交的交并比的功能，
# 最后的generate_anchor_base()的功能大概就是根据基准点生成9个基本的anchor的功能！
# ratios=[0.5,1,2],anchor_scales=[8,16,32]是长宽比和缩放比例，3x3的参数刚好得到9个anchor!



# 来看一下loc2bbox部分的代码，首先是一个if判断数据的类型，不是主要功能实现部分，紧接着src_height = src_bbox[:,2]- src_bbox[:,0]求出源框架的高度，
# 用[:,2]-[:,0]，之所以这么做是因为进行回归是要将数据格式从左上右下的坐标表示形式转化到中心点和长宽的表示形式，而bbox框的源位置类型应该是x0,y0,x1,y1这样用第三个减去第一个得到的自然是高度h，同样的办法也可以求出宽度w,
# 然后函数进行了中心点的求解，就是用左上角的x0+1/2h,y0+1/2w很直觉的就可以求出中心点的坐标，接下来利用 dy = loc[:,0::4],dx = loc[:,1::4],dh=loc[:,2::4],dw=loc[:,3::4]
# 分别求出回归预测loc的四个参数dy,dx,dh,dw来对源框bbox进行修正，即利用下述公式分别将源框的位置坐标转化为修正后框框的位置坐标x,y和宽度及高度w,h，
# 完成了目标框的位置确定，最后再将中心点坐标和长宽转换成左上角和右下角坐标的表示形式，就完成了loc2bbox函数的编写
# pic1.png
def loc2bbox(src_bbox, loc):
    """Decode bounding boxes from bounding box offsets and scales.

    Given bounding box offsets and scales computed by
    :meth:`bbox2loc`, this function decodes the representation to
    coordinates in 2D image coordinates.

    Given scales and offsets :math:`t_y, t_x, t_h, t_w` and a bounding
    box whose center is :math:`(y, x) = p_y, p_x` and size :math:`p_h, p_w`,
    the decoded bounding box's center :math:`\\hat{g}_y`, :math:`\\hat{g}_x`
    and size :math:`\\hat{g}_h`, :math:`\\hat{g}_w` are calculated
    by the following formulas.

    * :math:`\\hat{g}_y = p_h t_y + p_y`
    * :math:`\\hat{g}_x = p_w t_x + p_x`
    * :math:`\\hat{g}_h = p_h \\exp(t_h)`
    * :math:`\\hat{g}_w = p_w \\exp(t_w)`

    The decoding formulas are used in works such as R-CNN [#]_.

    The output is same type as the type of the inputs.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): A coordinates of bounding boxes.
            Its shape is :math:`(R, 4)`. These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        loc (array): An array with offsets and scales.
            The shapes of :obj:`src_bbox` and :obj:`loc` should be same.
            This contains values :math:`t_y, t_x, t_h, t_w`.

    Returns:
        array:
        Decoded bounding box coordinates. Its shape is :math:`(R, 4)`. \
        The second axis contains four values \
        :math:`\\hat{g}_{ymin}, \\hat{g}_{xmin},
        \\hat{g}_{ymax}, \\hat{g}_{xmax}`.

    """

    if src_bbox.shape[0] == 0:
        return xp.zeros((0, 4), dtype=loc.dtype)

    src_bbox = src_bbox.astype(src_bbox.dtype, copy=False)

    src_height = src_bbox[:, 2] - src_bbox[:, 0]
    src_width = src_bbox[:, 3] - src_bbox[:, 1]
    src_ctr_y = src_bbox[:, 0] + 0.5 * src_height
    src_ctr_x = src_bbox[:, 1] + 0.5 * src_width

    dy = loc[:, 0::4]
    dx = loc[:, 1::4]
    dh = loc[:, 2::4]
    dw = loc[:, 3::4]

    ctr_y = dy * src_height[:, xp.newaxis] + src_ctr_y[:, xp.newaxis]
    ctr_x = dx * src_width[:, xp.newaxis] + src_ctr_x[:, xp.newaxis]
    h = xp.exp(dh) * src_height[:, xp.newaxis]
    w = xp.exp(dw) * src_width[:, xp.newaxis]

    dst_bbox = xp.zeros(loc.shape, dtype=loc.dtype)
    dst_bbox[:, 0::4] = ctr_y - 0.5 * h
    dst_bbox[:, 1::4] = ctr_x - 0.5 * w
    dst_bbox[:, 2::4] = ctr_y + 0.5 * h
    dst_bbox[:, 3::4] = ctr_x + 0.5 * w

    return dst_bbox

# 这个函数的功能就是求出用于回归预测的ground_truth的值是多少，通俗点说就是你让我根据anchor来预测真实的目标的位置，那我需要学习，
# 我学习的过程你得给我一个用于计算损失函数的目标偏移量吧！没错，这个函数的作用就是用来计算这个目标偏移量的，bbox2loc,其计算遵循了下述公式：
# pic2.png
# pic3.png
# 仔细看代码你就会发现确实程序就是这样写的，首先同样的计算出源框架也就是预测框架的中心点坐标和它的长和宽，完成从左上右下表示的坐标的方式到中心点坐标表示方式的转化得到Px,Py,Pw,Ph，
# 同样的将ground_truth的左上右下角的坐标转换成中心点坐标和长宽的形式也就是上面公式里面的Gx,Gy,Gw,Gh，紧接着利用eps = xp.finfo(height.dtype).eps求出最小的正数，将height,width与其比较保证全部是非负！
# 之后就利用上述的公式求出偏移量的值tx,ty,tw,th完成了从bbox到loc的转化！
def bbox2loc(src_bbox, dst_bbox):
    """Encodes the source and the destination bounding boxes to "loc".

    Given bounding boxes, this function computes offsets and scales
    to match the source bounding boxes to the target bounding boxes.
    Mathematcially, given a bounding box whose center is
    :math:`(y, x) = p_y, p_x` and
    size :math:`p_h, p_w` and the target bounding box whose center is
    :math:`g_y, g_x` and size :math:`g_h, g_w`, the offsets and scales
    :math:`t_y, t_x, t_h, t_w` can be computed by the following formulas.

    * :math:`t_y = \\frac{(g_y - p_y)} {p_h}`
    * :math:`t_x = \\frac{(g_x - p_x)} {p_w}`
    * :math:`t_h = \\log(\\frac{g_h} {p_h})`
    * :math:`t_w = \\log(\\frac{g_w} {p_w})`

    The output is same type as the type of the inputs.
    The encoding formulas are used in works such as R-CNN [#]_.

    .. [#] Ross Girshick, Jeff Donahue, Trevor Darrell, Jitendra Malik. \
    Rich feature hierarchies for accurate object detection and semantic \
    segmentation. CVPR 2014.

    Args:
        src_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`. :math:`R` is the number of bounding boxes.
            These coordinates are
            :math:`p_{ymin}, p_{xmin}, p_{ymax}, p_{xmax}`.
        dst_bbox (array): An image coordinate array whose shape is
            :math:`(R, 4)`.
            These coordinates are
            :math:`g_{ymin}, g_{xmin}, g_{ymax}, g_{xmax}`.

    Returns:
        array:
        Bounding box offsets and scales from :obj:`src_bbox` \
        to :obj:`dst_bbox`. \
        This has shape :math:`(R, 4)`.
        The second axis contains four values :math:`t_y, t_x, t_h, t_w`.

    """

    height = src_bbox[:, 2] - src_bbox[:, 0]
    width = src_bbox[:, 3] - src_bbox[:, 1]
    ctr_y = src_bbox[:, 0] + 0.5 * height
    ctr_x = src_bbox[:, 1] + 0.5 * width

    base_height = dst_bbox[:, 2] - dst_bbox[:, 0]
    base_width = dst_bbox[:, 3] - dst_bbox[:, 1]
    base_ctr_y = dst_bbox[:, 0] + 0.5 * base_height
    base_ctr_x = dst_bbox[:, 1] + 0.5 * base_width

    eps = xp.finfo(height.dtype).eps
    height = xp.maximum(height, eps)
    width = xp.maximum(width, eps)

    dy = (base_ctr_y - ctr_y) / height
    dx = (base_ctr_x - ctr_x) / width
    dh = xp.log(base_height / height)
    dw = xp.log(base_width / width)

    loc = xp.vstack((dy, dx, dh, dw)).transpose()
    return loc

# 顾名思义，这个函数的作用就是计算两个bbox的IOU，所谓的IOU其实就是交并比，而 这个交并比就是两个IOU相交的面积除以相并的面积，用公式来表示就是：
# pic4.png
# 这样的表达应该足够直观了吧，而整个函数也正是按照这个思路进行的，来看代码首先不满足.shape[1]的判断，说明bbox的形状不完整，直接raise IndexError ，
# 然后分别取两个IOU左上的最大值和右下的最小值，这样其实就是完成了相交的工作(因为bbox现在的表示方式是左上坐标和右下坐标)，之后利用numpy.prod返回给定轴上数组元素的乘积，
# 分别求出area_i (相交的面积) area_a,area_b(两个bbox的面积)最后直接利用公式 area_i / area_a +area_b - area_i 就求出了两个框框之间的交并比!
def bbox_iou(bbox_a, bbox_b):
    """Calculate the Intersection of Unions (IoUs) between bounding boxes.

    IoU is calculated as a ratio of area of the intersection
    and area of the union.

    This function accepts both :obj:`numpy.ndarray` and :obj:`cupy.ndarray` as
    inputs. Please note that both :obj:`bbox_a` and :obj:`bbox_b` need to be
    same type.
    The output is same type as the type of the inputs.

    Args:
        bbox_a (array): An array whose shape is :math:`(N, 4)`.
            :math:`N` is the number of bounding boxes.
            The dtype should be :obj:`numpy.float32`.
        bbox_b (array): An array similar to :obj:`bbox_a`,
            whose shape is :math:`(K, 4)`.
            The dtype should be :obj:`numpy.float32`.

    Returns:
        array:
        An array whose shape is :math:`(N, K)`. \
        An element at index :math:`(n, k)` contains IoUs between \
        :math:`n` th bounding box in :obj:`bbox_a` and :math:`k` th bounding \
        box in :obj:`bbox_b`.

    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError

    # top left
    tl = xp.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = xp.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])

    area_i = xp.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = xp.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = xp.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)
    return area_i / (area_a[:, None] + area_b - area_i)


def __test():
    pass


if __name__ == '__main__':
    __test()

# 这个函数的作用就是产生(0,0)坐标开始的基础的9个anchor框，(0,0)坐标是指的一次提取的特征图而言，从函数的名字我们也可以看出来，generate_anchor_base,
# 分析一下函数的参数base_size=16就是基础的anchor的宽和高其实是16的大小，再根据不同的放缩比和宽高比进行进一步的调整，ratios就是指的宽高的放缩比分别是0.5:1,1:1,1:2这样，
# 最后一个参数是anchor_scales也就是在base_size的基础上再增加的量，本代码中对应着三种面积的大小(16*8)2 ,
# (16*16)2  (16*32)2  也就是128,256,512的平方大小，三种面积乘以三种放缩比就刚刚好是9种anchor，示意图如下：
# pic5.png
# 其实，Faster-rcnn的重要思想就是在这个地方体现出来了，到底怎样进行目标检测？如何才能不漏下任何一个目标？那就是遍历的方法，不是遍历图片，而是遍历特征图，
# 对一次提取的特征图进行遍历(3*3的卷积核挨个特征产生anchor) 依次产生9个长宽比尺寸不同的anchor，力求将所有的在图中的目标都框住，产生完anchor之后再送入到9×2和9×4的Fc网络用来做分类和回归，
# 对产生的anchor进行进一步的修正，这样几乎以极大的概率可以将图中所有的目标全部框住了！ 后续再进行一些处理，如非极大值抑制，抑制住重复框住的anchor，产生良好的可视效果！
def generate_anchor_base(base_size=16, ratios=[0.5, 1, 2],
                         anchor_scales=[8, 16, 32]):
    """Generate anchor base windows by enumerating aspect ratio and scales.

    Generate anchors that are scaled and modified to the given aspect ratios.
    Area of a scaled anchor is preserved when modifying to the given aspect
    ratio.

    :obj:`R = len(ratios) * len(anchor_scales)` anchors are generated by this
    function.
    The :obj:`i * len(anchor_scales) + j` th anchor corresponds to an anchor
    generated by :obj:`ratios[i]` and :obj:`anchor_scales[j]`.

    For example, if the scale is :math:`8` and the ratio is :math:`0.25`,
    the width and the height of the base window will be stretched by :math:`8`.
    For modifying the anchor to the given aspect ratio,
    the height is halved and the width is doubled.

    Args:
        base_size (number): The width and the height of the reference window.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    Returns:
        ~numpy.ndarray:
        An array of shape :math:`(R, 4)`.
        Each element is a set of coordinates of a bounding box.
        The second axis corresponds to
        :math:`(y_{min}, x_{min}, y_{max}, x_{max})` of a bounding box.

    """
    py = base_size / 2.
    px = base_size / 2.

    anchor_base = np.zeros((len(ratios) * len(anchor_scales), 4),
                           dtype=np.float32)
    for i in six.moves.range(len(ratios)):
        for j in six.moves.range(len(anchor_scales)):
            h = base_size * anchor_scales[j] * np.sqrt(ratios[i])
            w = base_size * anchor_scales[j] * np.sqrt(1. / ratios[i])

            index = i * len(anchor_scales) + j
            anchor_base[index, 0] = py - h / 2.
            anchor_base[index, 1] = px - w / 2.
            anchor_base[index, 2] = py + h / 2.
            anchor_base[index, 3] = px + w / 2.
    return anchor_base
