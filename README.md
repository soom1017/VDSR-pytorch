# VDSR-pytorch
VDSR(CVPR 2016) pytorch implementation

Implementation of CVPR2016 Paper: ["Accurate Image Super-Resolution Using Very Deep Convolutional Networks"](https://cv.snu.ac.kr/research/VDSR/VDSR_CVPR2016.pdf) in PyTorch

## Usage
### Training
```
python3 src/train.py
```

### Prepare Training dataset
- Training dataset consists of 291 images, 200 for BSDS200 and 91 for T91.
- Download dataset [here](http://vllab.ucmerced.edu/wlai24/LapSRN/), and make structure like below.

```sh
VDSR-pytorch
│ 
├─ LICENSE
├─ README.md
│ 
├─ data
│  └─ 291Trainset
│           img001.png
│           img002.png
│           ...
├─ outputs
├─ src
└─ utils
```

### Performance
- Trained my VSDR model on 291 images with data augmentation
- No bias is used in this implementation, and the gradient clipping's implementation is different from paper
