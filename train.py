
import os
os.environ['CUDA_PATH']='/usr/local/cuda-8.0/'
os.environ['LD_LIBRARY_PATH']='/usr/lib/nvidia-375:/usr/local/cuda-8.0/lib64'
import torch as t
from torch.utils import data as data_
from torch.autograd import Variable

from trainer import FasterRCNNTrainer
from data.dataset import Dataset
from config import opt
from tqdm import tqdm
import matplotlib 
matplotlib.use('agg')
from util import array_tool as at

from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from util.vis_tool import visdom_bbox



def update_meters(meters, loss_dict):
    for key,meter in meters.items():
        meter.add(loss_dict[key])

def reset_meters(meters):
    for key,meter in meters.items():
        meter.reset()
def get_data(meters):
    return {k:v.value()[0] for k,v in meters.items()}


def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('load data')
    dataloader = data_.DataLoader(dataset,\
                            batch_size=1,\
                            shuffle=True,\
                            num_workers=opt.num_workers)

    faster_rcnn = FasterRCNNVGG16()
    from torchnet.meter import AverageValueMeter
    print('model completed')
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load_state_dict(t.load(opt.load_path))
        print('load pretrained model from %s' %opt.load_path)
    
    trainer.vis.text(dataset.db.label_names,win='labels')
    

    meters = None
    for epoch in range(opt.epoch):
        trainer.rpn_cm.reset()
        trainer.roi_cm.reset()
        for ii,(img, bbox_, label_, scale, ori_img) in tqdm(enumerate(dataloader),total=len(dataset)):
            scale = at.scalar(scale) #[0]
            img,bbox,label = img.cuda().float(),bbox_.cuda(),label_.cuda()
            img,bbox,label = Variable(img),Variable(bbox),Variable(label)
            losses,rois = trainer.train_step(img,bbox,label,scale)
            loss_d = {k:at.scalar(v) for k,v in losses._asdict().items()}
            if meters is None:
                meters = {k:AverageValueMeter() for k in loss_d}
            update_meters(meters,loss_d)
            if (ii)%opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    import ipdb;ipdb.set_trace()
                trainer.vis.plot_many(get_data(meters))
                ori_img_ =  (img*0.225+0.45).clamp(min=0,max=1)*255
                trainer.vis.img('train_data',visdom_bbox(at.tonumpy(ori_img_)[0],at.tonumpy(bbox_)[0],label_[0].numpy()))
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict(ori_img)
                trainer.vis.img('pp',visdom_bbox(at.tonumpy(ori_img[0]),at.tonumpy(_bboxes[0]),at.tonumpy(_labels[0]).reshape(-1)))
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()),win='rpn_cm')
                a2c_ = trainer.roi_cm.value()
                a2c_[1:,1:] = 0.2 * a2c_[1:,1:]
                # trainer.vis.text(str(trainer.roi_cm.value().tolist()),win='roi_cm')
                trainer.vis.img('roi_cm',at.totensor(a2c_/a2c_.max(),False).float())
                trainer.vis.img('roi-top4',visdom_bbox((at.tonumpy(img[0])*0.25+0.45).clip(min=0,max=1)*255,at.tonumpy(rois[:4])))
        reset_meters(meters)
        if epoch==1:trainer.faster_rcnn.update_optimizer(1e-4,1e-4,1e-4)
        t.save(trainer.state_dict(),'/mnt/3/faster_%s.pth' %epoch)

if __name__=='__main__':
    import fire
    fire.Fire()