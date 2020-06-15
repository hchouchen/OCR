# Deep learning -- OCR 


## introduction
交叉项目综合训练 视觉组"深度学习文字识别"组工作。

input: 经过区域提取的含有文字的图片

output: 识别结果,type=str

model: CRNN+CTC Loss, 详见项目报告

## Requirement
python>=3.6  
pytorch-cuda  
PIL  
sklearn  
copy,os,numpy,pandas,matplotlib,argparse

## CONFIG
The following script is for training and testing:

`python main.py CONFIG [--mode 'train' or 'test' default='train'] [--batch_size BATCH_SIZE default=64] [--language LANGUAGE_TYPE default='Russian'] [--pretrained WHETHER TO USE PRETRAINED WEIGHT(if mode=='test': MUST TRUE) default='true'] [--weight_path MODEL_WEIGHT_PATH default=r'./Russian-weight.pth'] [--imgH IMAGE HEIGHT default=32] [--img_path IMG .NPY PATH default=r'./Russian-data.npy'] [--label_path LABEL .CSV PATH default=r'./Russian-label.csv'] [--epoch MAX EPOCH default=10] [--test_root TEST IMG ROOTPATH default=r'./test']`

### example: Train
`python main.py --mode 'train'  --batch_size 64  --language 'Russian'  --pretrained false  --imgH 32  --img_path './data.npy'  --label_path './label.csv'  --epoch 20`

output:  
echo: 0  
train-loss: 1.2  
test-loss: 1.5  
accuracy: 0.52  
...

### example: Test
`python main.py --mode 'test'  --pretrained true  --weight_path './weight.pth'  --test_root './test_img'`

output:  
result str list