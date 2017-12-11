import os

import ipdb
import matplotlib
from tqdm import tqdm

import torch as t
from config import opt
from data.dataset import Dataset
from model import FasterRCNNVGG16
from torch.autograd import Variable
from torch.utils import data as data_
from trainer import FasterRCNNTrainer
from util import array_tool as at
from util.vis_tool import visdom_bbox

matplotlib.use('agg')

def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,\
                            batch_size=1,\
                            shuffle=True,\
                            num_workers=opt.num_workers)

    faster_rcnn = FasterRCNNVGG16()
    print('model completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load_state_dict(t.load(opt.load_path))
        print('load pretrained model from %s' %opt.load_path)
    
    trainer.vis.text(dataset.db.label_names,win='labels')

    for epoch in range(opt.epoch):
        trainer.reset_meters()
        for ii,(img, bbox_, label_, scale, ori_img) in tqdm(enumerate(dataloader),total=len(dataset)):
            scale = at.scalar(scale)
            img,bbox,label = img.cuda().float(),bbox_.cuda(),label_.cuda()
            img,bbox,label = Variable(img),Variable(bbox),Variable(label)
            losses,rois = trainer.train_step(img,bbox,label,scale)
            loss_d = {k:at.scalar(v) for k,v in losses._asdict().items()}
            
            if (ii+1)%opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()
                trainer.vis.plot_many(trainer.get_meter_data())
                ori_img_ =  (img*0.225+0.45).clamp(min=0,max=1)*255
                trainer.vis.img('train',visdom_bbox(at.tonumpy(ori_img_)[0],at.tonumpy(bbox_)[0],label_[0].numpy()))
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(ori_img)
                trainer.vis.img('predict',visdom_bbox(at.tonumpy(ori_img[0]),at.tonumpy(_bboxes[0]),at.tonumpy(_labels[0]).reshape(-1)))
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()),win='rpn_cm')
                a2c_ = trainer.roi_cm.value().copy()
                a2c_[1:,1:] = a2c_[1:,1:]*10
                # trainer.vis.text(str(trainer.roi_cm.value().tolist()),win='roi_cm')
                trainer.vis.img('roi_cm',at.totensor(a2c_,False).float())
                trainer.vis.img('roi_top4',visdom_bbox((at.tonumpy(img[0])*0.25+0.45).clip(min=0,max=1)*255,at.tonumpy(rois[:4])))
        # if epoch==1:trainer.faster_rcnn.update_optimizer(1e-4,1e-4,1e-4)
        if epoch==2: trainer.faster_rcnn.update_optimizer(0,5e-4,1e-3)
        if epoch==5: trainer.faster_rcnn.update_optimizer(2e-4,5e-4,5e-4)
        if epoch==12: trianer.faster_rcnn.update_optimizer(1e-4,1e-4,1e-4)
        t.save(trainer.state_dict(),'checkpoints/fasterrcnn_%s.pth' %epoch)


if __name__=='__main__':
    import fire
    fire.Fire()
