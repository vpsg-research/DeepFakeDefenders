# The Global Multimedia Deepfake Detection Challenge

<center><img src="competition.png "width="100%"></center>

We won the Top 20 Excellence Awards in the Global Multimedia Deepfake Detection. Our team consists of Lixin Jia, Hongrui Zheng and Zhiqing Guo([www.guozhiqing.cn](https://www.guozhiqing.cn/)).

# Our Competition Model
Enhanced EfficientNet for Face Forgery Detection Using Custom Activations and Augmentation.

## Setup
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

Or you can use the requirement.txt file to install the necessary packages.

## Device
Single NVIDIA GeForce RTX 4090

## Run
#### Step 1. Prepare the Datasets

The dataset should contain image files and txt files of the corresponding images and labels, including the training set, validation set, and test set.

#### Step 2. Quick Start

## Training

```bash
python train.py --trainset_label_path '/your/path/trainset_label.txt' --valset_label_path '/your/path/valset_label.txt' --trainset_path '/your/path/trainset/' --valset_path '/your/path/valset/'
```
For example, if run on the below path:
```bash
python train.py --trainset_label_path '/root/autodl-tmp/Competition/competition/trainset_label.txt' --valset_label_path '/root/autodl-tmp/Competition/competition/valset_label.txt' --trainset_path '/root/autodl-tmp/phase1/trainset/' --valset_path '/root/autodl-tmp/phase1/valset/'
```

## Testing
Choose the best model generated during training:
```bash
python test.py --testset_label_path '/your/path/testset1_seen_nolabel.txt'  --testset_path '/your/path/testset1_seen/' --model_path '/your/path//model_.pt'
```

For example, if you want test on model_95.48:
```bash
python test.py --testset_label_path '/root/autodl-tmp/phase2/testset1_seen_nolabel.txt'  --testset_path '/root/autodl-tmp/phase2/testset1_seen/' --model_path './model_95.48.pt'
```
