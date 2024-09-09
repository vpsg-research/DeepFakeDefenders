import torch
torch.manual_seed(0)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
import argparse
import torchvision.transforms as transforms
from torch.utils.data.dataset import Dataset
import timm
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from EMixnet import EMixnet

# Transform the test data 5 predictions
def predict(test_loader, model, tta=5):
    # switch to evaluate mode
    model.eval()
    
    test_pred_tta = None
    for _ in range(tta):
        test_pred = []
        with torch.no_grad():
            for i, (input) in tqdm(enumerate(test_loader), total=len(test_loader)):
                input = input.cuda()

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
    
    # Apply threshold to get binary predictions
    #test_pred_tta = (test_pred_tta > 0.5).astype(int)
    
    return test_pred_tta

class FFDIDataset(Dataset):
    def __init__(self, img_path, transform=None):
        self.img_path = img_path
        
        if transform is not None:
            self.transform = transform
        else:
            self.transform = None
    
    def __getitem__(self, index):
        img = Image.open(self.img_path[index]).convert('RGB')
        
        if self.transform is not None:
            img = self.transform(img)
        
        return img
    
    def __len__(self):
        return len(self.img_path)

def main():
    args = parse.parse_args()
    testset_label = args.testset_label_path
    testset = args.testset_path
    test_label = pd.read_csv(testset_label)

    test_label['path'] = testset + test_label['img_name']
    # loading model
    base_model = timm.create_model('tf_efficientnet_b3_ns', pretrained=True, num_classes=0)
    base_model = base_model.cuda()

    model = EMixnet(base_model)
    model_path = args.model_path
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()

    test_loader = torch.utils.data.DataLoader(
        FFDIDataset(test_label['path'], 
                transforms.Compose([
                            transforms.Resize((300, 300)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        ), batch_size=32, shuffle=False, num_workers=4, pin_memory=True
    )

    test_final_prediction = predict(test_loader, model, 5)
    print(test_final_prediction)
    test_label['y_pred'] = test_final_prediction[:, 0]
    test_label[['img_name', 'y_pred']].to_csv('submission_test.csv', index=None)

if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--testset_label_path', '-tlp', type=str, default = '/root/autodl-tmp/phase2/testset1_seen_nolabel.txt')
    parse.add_argument('--testset_path', '-tp', type=str, default = '/root/autodl-tmp/phase2/testset1_seen/')
    parse.add_argument('--model_path', '-mp', type=str, default='./model_96.67.pt')
    
    main()