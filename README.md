# VDSR-pytorch
VDSR(CVPR 2016) pytorch implementation

Implementation of CVPR2016 Paper: ["Accurate Image Super-Resolution Using Very Deep Convolutional Networks"](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf) in PyTorch

### Prepare Training dataset
- Download train dataset with hdf5 format from [here](https://github.com/twtygqyy/pytorch-vdsr).

### Performance
- Trained my VDSR model on 291 images with data augmentation
- No bias is used in this implementation, and the gradient clipping's implementation is different from paper
