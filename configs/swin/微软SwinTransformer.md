## 微软Swin Transformer 

### Github
https://github.com/microsoft/Swin-Transformer/

### 图片分类
── mydataset
  ├── train_map.txt
  ├── train.zip
  ├── val_map.txt
  └── val.zip

mydataset/train.zip
  ├── imgs/aa01.JPEG
  ├── imgs/aa02.JPEG

mydataset/train_map.txt
imgs/aa01.JPEG    65
imgs/aa02.JPEG   970

文本用\t分隔,不是空格或逗号

python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune.yaml --data-path mydataset --zip --batch-size 4 --output output --opts TRAIN.EPOCHS 20 MODEL.NUM_CLASSES 4


python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345  main.py --cfg configs/swin/swin_tiny_patch4_window7_224_22kto1k_finetune.yaml --data-path /media/hello/453C61C6A3A805B6/AI/dataset/22_06_led/LED_part1_train/swin_dataset --batch-size 8 --output output --zip --opts TRAIN.EPOCHS 20 MODEL.NUM_CLASSES 4