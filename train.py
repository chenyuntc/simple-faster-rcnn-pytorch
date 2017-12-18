import os

import ipdb
import matplotlib
from tqdm import tqdm

import torch as t
from config import opt
from data.dataset import Dataset,TestDataset
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from util import array_tool as at
from util.vis_tool import visdom_bbox
from util.eval_tool import eval_detection_voc

matplotlib.use('agg')


def eval(dataloader,faster_rcnn,test_num=1000):
    pred_bboxes,pred_labels, pred_scores = list(),list(),list()
    gt_bboxes,gt_labels, gt_difficults = list(),list(),list()
    for ii,(imgs,sizes,gt_bboxes_,gt_labels_,gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0],sizes[1][0]]
        pred_bboxes_,pred_labels_,pred_scores_ = faster_rcnn.predict2(imgs,[sizes])
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        if ii==test_num:break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    return result

def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,\
                            batch_size=1,\
                            shuffle=True,\
                            # pin_memory=True,
                            num_workers=opt.num_workers)
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(testset,
                                batch_size=1,
                                num_workers=2,
                                shuffle=True,\
                                # pin_memory=True
                                )

    faster_rcnn = FasterRCNNVGG16()
    print('model completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load_state_dict(t.load(opt.load_path))
        print('load pretrained model from %s' %opt.load_path)
    
    trainer.vis.text(dataset.db.label_names,win='labels')

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii,(img, bbox_, label_, scale, ori_img) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img,bbox,label = img.cuda().float(),bbox_.cuda(),label_.cuda()
            img,bbox,label = Variable(img),Variable(bbox),Variable(label)
            losses = trainer.train_step(img,bbox,label,scale)
            
            if (ii+1)%opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                # plot loss
                trainer.vis.plot_many(trainer.get_meter_data())
                
                # plot groud truth bboxes
                ori_img_ =  (img*0.225+0.45).clamp(min=0,max=1)*255
                trainer.vis.img('gt_img',visdom_bbox(at.tonumpy(ori_img_)[0],at.tonumpy(bbox_)[0],label_[0].numpy()))
                
                # plot predicti bboxes
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(ori_img)
                trainer.vis.img('pred_img',visdom_bbox(at.tonumpy(ori_img[0]),at.tonumpy(_bboxes[0]),at.tonumpy(_labels[0]).reshape(-1),at.tonumpy(_scores[0])))

                # rpn confusion matrix(meter)
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()),win='rpn_cm')
                # roi confusion matrix
                trainer.vis.img('roi_cm',at.totensor(trainer.roi_cm.value(),False).float())

                # ooo_ = (at.tonumpy(img[0])*0.25+0.45).clip(min=0,max=1)*255
                # trainer.vis.img('rpn_roi_top4',
                #                     visdom_bbox(ooo_,
                #                     at.tonumpy(rois[:4]))
                #             )
                # trainer.vis.img('sample_rois_img', 
                #         visdom_bbox(ooo_,
                #             at.tonumpy(trainer.sample_roi[0:12:2]),
                #             trainer.gt_roi_label[0:12:2]-1)
                #             )
                # break #TODO:delete it for debug
        if epoch==6: # lr decay
            trainer.faster_rcnn.update_optimizer(opt.lr_decay)

        eval_result  = eval(test_dataloader,faster_rcnn)
        trainer.vis.plot('test_map', eval_result['map'])
        trainer.vis.log('map:{},loss:{},roi_cm:{}'.format(str(eval_result),str(trainer.get_meter_data()),str(trainer.rpn_cm.conf.tolist())))
        trainer.save()
        # t.save(trainer.state_dict(),'checkpoints/fasterrcnn_%s.pth' %epoch)
        # t.vis.save([opt.env])


if __name__=='__main__':
    import fire
    fire.Fire()
