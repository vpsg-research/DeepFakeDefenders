import torch
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import argparse
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataset import Dataset
import timm
import time
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import numpy as np
from EMixnet import EMixnet
from post_process import process_images, append_labels

class AverageMeter(object):
        """Computes and stores the average and current value"""
        def __init__(self, name, fmt=':f'):
            self.name = name
            self.fmt = fmt
            self.reset()

        def reset(self):
            self.val = 0
            self.avg = 0
            self.sum = 0
            self.count = 0

        def update(self, val, n=1):
            self.val = val
            self.sum += val * n
            self.count += n
            self.avg = self.sum / self.count

        def __str__(self):
            fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
            return fmtstr.format(**self.__dict__)

class ProgressMeter(object):
    def __init__(self, num_batches, *meters):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = ""


    def pr2int(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'
    
def validate(val_loader, model, criterion):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')

    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (input, target) in tqdm(enumerate(val_loader), total=len(val_loader)):
            input = input.cuda()
            target = target.cuda().float().unsqueeze(1)

            output = model(input)
            loss = criterion(output, target)

            predicted = (output > 0.5).float()
            acc = (predicted == target).float().mean() * 100
            losses.update(loss.item(), input.size(0))
            top1.update(acc, input.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

        print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
        return top1.avg

def train(train_loader, model, criterion, optimizer, epoch):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, losses, top1)

    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda().float().unsqueeze(1)  # 调整目标形状

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        predicted = (output > 0.5).float()
        acc = (predicted == target).float().mean() * 100
        top1.update(acc, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            progress.pr2int(i)

class FFDIDataset(Dataset):
    def __init__(self, img_path, img_label, transform=None):
        self.img_path = img_path
        self.img_label = img_label
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img, torch.from_numpy(np.array(self.img_label[index]))
    
    def __len__(self):
        return len(self.img_path)

# Randomly occludes areas of the image
def random_erase(image, probability=0.1, sl=0.1, sh=0.2, r1=0.5):
    if np.random.rand() > probability:
        return image

    c, h, w = image.shape
    area = h * w

    for _ in range(100):
        target_area = np.random.uniform(sl, sh) * area
        aspect_ratio = np.random.uniform(r1, 1/r1)

        h_erase = int(round(np.sqrt(target_area * aspect_ratio)))
        w_erase = int(round(np.sqrt(target_area / aspect_ratio)))

        if h_erase < h and w_erase < w:
            x1 = np.random.randint(0, h - h_erase)
            y1 = np.random.randint(0, w - w_erase)
            image[:, x1:x1 + h_erase, y1:y1 + w_erase] = torch.rand(c, h_erase, w_erase)
            return image

    return image

def predict(test_loader, model, tta=10):
        # switch to evaluate mode
        model.eval()
        
        test_pred_tta = None
        for _ in range(tta):
            test_pred = []
            with torch.no_grad():
                for i, (input, target) in tqdm(enumerate(test_loader), total=len(test_loader)):
                    input = input.cuda()
                    target = target.cuda()

                    # compute output
                    output = model(input)
                    output = output.data.cpu().numpy()

                    test_pred.append(output)
            test_pred = np.vstack(test_pred)
        
            if test_pred_tta is None:
                test_pred_tta = test_pred
            else:
                test_pred_tta += test_pred
        
        # Average the predictions if TTA is used
        test_pred_tta /= tta
        return test_pred_tta

def main():
    args = parse.parse_args()
    trainset_label = args.trainset_label_path
    valset_label = args.valset_label_path
    trainset = args.trainset_path
    valset = args.valset_path
    out_file = process_images(trainset_label, trainset, trainset)
    append_labels(out_file, trainset_label)

    train_label = pd.read_csv(trainset_label)
    val_label = pd.read_csv(valset_label)

    train_label['path'] = trainset + train_label['img_name']
    val_label['path'] = valset + val_label['img_name']

    # loading model
    base_model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=0)
    base_model = base_model.cuda()

    epoch_num = 5
    bs_value = 32

    model = EMixnet(base_model)
    model = model.cuda()

    train_loader = torch.utils.data.DataLoader(
        FFDIDataset(train_label['path'], train_label['target'],             
                transforms.Compose([
                            transforms.Resize((300, 300)),
                            transforms.ToTensor(),
                            transforms.Lambda(lambda x: random_erase(x)),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=bs_value, shuffle=True, num_workers=4, pin_memory=True
    )

    val_loader = torch.utils.data.DataLoader(
        FFDIDataset(val_label['path'], val_label['target'], 
                transforms.Compose([
                            transforms.Resize((300, 300)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=bs_value, shuffle=False, num_workers=4, pin_memory=True
    )
    criterion = nn.BCELoss().cuda()
    optimizer = torch.optim.Adam(model.parameters(), 0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.85)
    best_acc = 0.0
    for epoch in range(epoch_num):
        scheduler.step()
        print('Epoch: ', epoch)

        train(train_loader, model, criterion, optimizer, epoch)
        val_acc = validate(val_loader, model, criterion)
        val_acc = val_acc.item()  
        if val_acc > best_acc:
            best_acc = round(val_acc, 2)
            torch.save(model.state_dict(), f'./model_{best_acc}.pt')

    print(f'Best validation accuracy: {best_acc}')

    val_label['y_pred'] = predict(val_loader, model, tta=1)[:, 0]  
    val_label[['img_name', 'y_pred']].to_csv('submission_train.csv', index=None)

    df = pd.read_csv('./submission_train.csv')
    true = {"img_name":[],"y_pred":[]}
    with open(valset_label, 'r', encoding='utf-8') as file:
        for line in file:
            if line.strip().split(',')[0] == 'img_name':
                continue
            true["img_name"].append(line.strip().split(',')[0])
            true["y_pred"].append(float(line.strip().split(',')[1]))

    true_df = pd.DataFrame(true)

    df_merged = pd.merge(df, true_df, on='img_name')
    df_merged
    y_pred = df_merged['y_pred_x']
    y_true = df_merged['y_pred_y']

    fpr, tpr, thresholds = roc_curve(y_true, y_pred)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.10f}')

    target_fpr = 1e-3
    tpr_at_target_fpr = np.interp(target_fpr, fpr, tpr)
    print(f'TPR at FPR=1E-3: {tpr_at_target_fpr:.10f}')

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--trainset_label_path', '-tlp', type=str, default = '/root/autodl-tmp/Competition/competition/trainset_label.txt')
    parse.add_argument('--valset_label_path', '-vlp', type=str, default = '/root/autodl-tmp/Competition/competition/valset_label.txt')
    parse.add_argument('--trainset_path', '-tp', type=str, default = '/root/autodl-tmp/phase1/trainset/')
    parse.add_argument('--valset_path', '-vp', type=str, default = '/root/autodl-tmp/phase1/valset/')

    main()