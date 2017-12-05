
from collections import namedtuple
from string import Template

import chainer.functions as F

import cupy as cp
import torch as t
from pynvrtc.compiler import Program

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'

@cupy.util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)
CUDA_NUM_THREADS = 1024
def GET_BLOCKS(N, K=CUDA_NUM_THREADS):
    return (N + K - 1) // K


forward_kernel = '''
extern "C"
__global__ void roi_forward(const float* const bottom_data,const float* const bottom_rois,
            float* top_data, int* argmax_data,
            const double spatial_scale,const int channels,const int height, 
            const int width, const int pooled_height, 
            const int pooled_width,const int NN
){
    
int idx = blockIdx.x * blockDim.x + threadIdx.x;
//printf("%d,%d,%d,%d  ", blockIdx.x, blockDim.x,threadIdx.x,i);
//printf("%d-" ,NN);
if(idx>NN)
    return;
const int pw = idx % pooled_width;
const int ph = (idx / pooled_width) % pooled_height;
const int c = (idx / pooled_width / pooled_height) % channels;
int num = idx / pooled_width / pooled_height / channels;
const int roi_batch_ind = bottom_rois[num * 5 + 0];
const int roi_start_w = round(bottom_rois[num * 5 + 1] * spatial_scale);
const int roi_start_h = round(bottom_rois[num * 5 + 2] * spatial_scale);
const int roi_end_w = round(bottom_rois[num * 5 + 3] * spatial_scale);
const int roi_end_h = round(bottom_rois[num * 5 + 4] * spatial_scale);
//printf("-%f-",spatial_scale);
//printf("%f,%f,%d,%d,%d  ",bottom_rois[num * 5 + 3],bottom_rois[num * 5 + 2] * spatial_scale,round(bottom_rois[num * 5 + 3] * spatial_scale),num,num*5+3);
//printf("-%d,%d,%d,%d-  ",roi_start_w,roi_start_h,roi_end_w,roi_end_h);
// Force malformed ROIs to be 1x1
const int roi_width = max(roi_end_w - roi_start_w + 1, 1);
const int roi_height = max(roi_end_h - roi_start_h + 1, 1);
const float bin_size_h = static_cast<float>(roi_height)
                / static_cast<float>(pooled_height);
const float bin_size_w = static_cast<float>(roi_width)
                / static_cast<float>(pooled_width);

int hstart = static_cast<int>(floor(static_cast<float>(ph)
                                * bin_size_h));
    int wstart = static_cast<int>(floor(static_cast<float>(pw)
                                * bin_size_w));
int hend = static_cast<int>(ceil(static_cast<float>(ph + 1)
                            * bin_size_h));
    int wend = static_cast<int>(ceil(static_cast<float>(pw + 1)
                            * bin_size_w));

// Add roi offsets and clip to input boundaries
hstart = min(max(hstart + roi_start_h, 0), height);
hend = min(max(hend + roi_start_h, 0), height);
wstart = min(max(wstart + roi_start_w, 0), width);
wend = min(max(wend + roi_start_w, 0), width);
bool is_empty = (hend <= hstart) || (wend <= wstart);

// Define an empty pooling region to be zero
float maxval = is_empty ? 0 : -1E+37;
// If nothing is pooled, argmax=-1 causes nothing to be backprop'd
int maxidx = -1;
const int data_offset = (roi_batch_ind * channels + c) * height * width;
for (int h = hstart; h < hend; ++h) {
    for (int w = wstart; w < wend; ++w) {
        int bottom_index = h * width + w;
        if (bottom_data[data_offset + bottom_index] > maxval) {
            maxval = bottom_data[data_offset + bottom_index];
            maxidx = bottom_index;
        }
    }
}
top_data[idx]=maxval;
argmax_data[idx]=maxidx;
//printf("%d,%d,%d,%d  ",pw,ph,num,c);
//printf("%d,%d,%f,%f  ",wstart-wend,roi_width,bin_size_h,roi_start_h);
//printf("%d,%d,%d,%d  ",roi_start_w,roi_start_h,roi_end_w,roi_end_h);
// }
}'''

backward_kernel='''
__global__ void roi_backward(const float* const top_diff, const int* const argmax_data, const int num_rois,
    const double spatial_scale, int channels, int height, int width,
    int pooled_height, int pooled_width,const float* const bottom_rois,float* bottom_diff)

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx>NN)
       return;

    int w = i % width;
    int h = (i / width) % height;
    int c = (i / (width * height)) % channels;
    int num = i / (width * height * channels);

    float gradient = 0;
    // Accumulate gradient over all ROIs that pooled this element
    for (int roi_n = 0; roi_n < num_rois; ++roi_n) {
        // Skip if ROI's batch index doesn't match num
        if (num != static_cast<int>(bottom_rois[roi_n * 5])) {
            continue;
        }

        int roi_start_w = round(bottom_rois[roi_n * 5 + 1]
                                * spatial_scale);
        int roi_start_h = round(bottom_rois[roi_n * 5 + 2]
                                * spatial_scale);
        int roi_end_w = round(bottom_rois[roi_n * 5 + 3]
                              * spatial_scale);
        int roi_end_h = round(bottom_rois[roi_n * 5 + 4]
                              * spatial_scale);

        // Skip if ROI doesn't include (h, w)
        const bool in_roi = (w >= roi_start_w && w <= roi_end_w &&
                             h >= roi_start_h && h <= roi_end_h);
        if (!in_roi) {
            continue;
        }

        int offset = (roi_n * channels + c) * pooled_height
                     * pooled_width;

        // Compute feasible set of pooled units that could have pooled
        // this bottom unit

        // Force malformed ROIs to be 1x1
        int roi_width = max(roi_end_w - roi_start_w + 1, 1);
        int roi_height = max(roi_end_h - roi_start_h + 1, 1);

        float bin_size_h = static_cast<float>(roi_height)
                       / static_cast<float>(pooled_height);
        float bin_size_w = static_cast<float>(roi_width)
                       / static_cast<float>(pooled_width);

        int phstart = floor(static_cast<float>(h - roi_start_h)
                            / bin_size_h);
        int phend = ceil(static_cast<float>(h - roi_start_h + 1)
                         / bin_size_h);
        int pwstart = floor(static_cast<float>(w - roi_start_w)
                            / bin_size_w);
        int pwend = ceil(static_cast<float>(w - roi_start_w + 1)
                         / bin_size_w);

        phstart = min(max(phstart, 0), pooled_height);
        phend = min(max(phend, 0), pooled_height);
        pwstart = min(max(pwstart, 0), pooled_width);
        pwend = min(max(pwend, 0), pooled_width);

        for (int ph = phstart; ph < phend; ++ph) {
            for (int pw = pwstart; pw < pwend; ++pw) {
                int index_ = ph * pooled_width + pw + offset;
                if (argmax_data[index_] == (h * width + w)) {
                    gradient += top_diff[index_];
                }
            }
        }
    }
    bottom_diff[idx] = gradient;
  
)(gy[0], self.argmax_data, bottom_rois.shape[0], self.spatial_scale,
  channels, height, width, self.outh, self.outw,
  bottom_rois, bottom_diff)
'''

cupy.cuda.runtime.free(0)

f_b = load_kernel('roi_backward',backward_kernel)
f=load_kernel('roi_forward',forward_kernel)
B,N,C,H,W,PH,PW = 2,8,4,32,32,7,7

bottom_data = t.randn(B,C,H,W).cuda()
bottom_rois = t.randn(N,5)
bottom_rois[:int(N/2),0]=0
bottom_rois[int(N/2):,0]=1
bottom_rois[:,1:] = (t.rand(N,4)*100).float()
bottom_rois = bottom_rois.cuda()
top_data = t.zeros(N,C,PH,PW).cuda()
argmax_data = t.zeros(N,C,PH,PW).cuda().int()
spatial_scale = 1./16
channels,height,width,pooled_height,pooled_width =\
C,H,W,PH,PW

bottom_diff = bottom_data.new(bottom_data.size()).fill_(0)
top_diff = top_data.new(top_data.size()).fill_(0)

##NOTE: python float 其实是c中的double
# f(args=[bottom_data.data_ptr(),bottom_rois.data_ptr(),
# top_data.data_ptr(),argmax_data.data_ptr(),
# spatial_scale,C,H,W,PH,PW,top_data.numel()],
# block=(CUDA_NUM_THREADS,1,1),
# grid=(GET_BLOCKS(top_data.numel()),1,1),
# stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

x=cp.array(bottom_data.cpu().numpy())
rois=cp.array(bottom_rois.cpu().numpy())
outh=PH
outw =PW

# cp_result = F.roi_pooling_2d(x, rois, outh, outw, spatial_scale)

cproi = F.ROIPooling2D(outh, outw, spatial_scale)
cp_result2=cproi.forward_gpu((x,rois))
aa = cp.asnumpy(cp_result2[0])
bb = top_data.cpu().numpy()
neq = (aa!=bb).sum()
assert neq==0,'output failed'
bb=argmax_data.cpu().numpy()
aa= cp.asnumpy(cproi.argmax_data)
neq = (aa!=bb).sum()
assert neq==0,'argmax failed'
