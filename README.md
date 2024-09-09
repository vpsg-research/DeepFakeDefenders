# The Global Multimedia Deepfake Detection Challenge

<center><img src="competition.png "width="100%"></center>

We won the Top 20 Excellence Awards in the Global Multimedia Deepfake Detection. Our team consists of Lixin Jia, Hongrui Zheng and [Dr. Zhiqing Guo](https://www.guozhiqing.cn/).

# Competition Model
Enhanced EfficientNet for Face Forgery Detection Using Custom Activations and Augmentation.

## Experimental Environment
- python==3.8.19
- numpy==1.24.4
- opencv-python==4.10.0.84
- pandas==2.0.3
- pillow==10.4.0
- scikit-learn==1.3.2
- scipy==1.10.1
- timm==0.5.4
- torch==2.4.0
- torchstat==0.0.7
- torchvision==0.19.0
- tqdm==4.66.4

Or you can use the requirement.txt file to install the necessary packages. We only use a single NVIDIA GeForce RTX 4090 for training.

## Quick Start

### Training

```bash
python train.py --trainset_label_path '/your/path/trainset_label.txt' --valset_label_path '/your/path/valset_label.txt' --trainset_path '/your/path/trainset/' --valset_path '/your/path/valset/'
```

### Testing
Choose the best model generated during training:
```bash
python test.py --testset_label_path '/your/path/testset1_seen_nolabel.txt'  --testset_path '/your/path/testset1_seen/' --model_path '/your/path/model_.pt'
```

### Pre-training Model
[Download](https://drive.google.com/file/d/1aZ1Qg1Yt2WGk9LbwVpQc-Ytp-5q49GvW/view?usp=sharing)
