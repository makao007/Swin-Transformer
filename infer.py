import os
import glob
import argparse
from PIL import Image

import torch
from torchvision import datasets

from config import get_config
from models import build_model
from data.build import build_transform

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer infer script', add_help=False)
    parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file')
    parser.add_argument('--weight', type=str, required=True, metavar="FILE", help='path to weight file')
    parser.add_argument('--data_path', type=str, required=True, metavar='PATH', help='the input image folder')
    parser.add_argument('--output', default='output', type=str, metavar='PATH', help='root of output folder, default: output')
    parser.add_argument("--device", type=str, default="cpu", help='device, cpu for cuda:0')
    parser.add_argument( "--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+', )
    args = parser.parse_args()
    args.zip = False
    args.cache_mode = False
    args.batch_size = 1
    args.pretrained = ''
    args.resume = False
    args.accumulation_steps = 1
    args.use_checkpoint = False
    args.amp_opt_level = '0'
    args.disable_amp = False
    args.tag = ''
    args.eval = True
    args.throughput = False
    args.local_rank = '0'
    return args

def main():
    args = parse_option()

    config = get_config(args)

    transform = build_transform (False, config)
    # dataset = datasets.ImageFolder(os.path.join(os.getcwd(), args.data_path), transform)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    device = torch.device(args.device)
    model = build_model(config).to(device)
    model.load_state_dict(torch.load(args.weight, map_location='cpu')['model'], strict=True)
    model.eval()
    # with torch.no_grad():
    #     for imgs, _ in dataloader:
    #         predicted = model(imgs.to(device))
    #         print (predicted)

    with torch.no_grad():
        files = glob.glob(os.path.join(os.getcwd(), args.data_path) + '/*')
        result = []
        for f in files:
            predicted = model(transform(Image.open(f).convert('RGB')).to(device).unsqueeze(dim=0))
            categories= torch.argmax(predicted, dim=1)
            result.append(os.path.split(f)[-1] + '\t' + str(categories[0].item()))
        
        output_file = args.output
        if not output_file.endswith('.txt'):
            output_file = os.path.join(output_file, 'output.txt')
        if not os.path.isdir(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file))
        with open(output_file, 'w', encoding='utf-8') as w:
            w.write('\n'.join(result))
        print ("save result to ", output_file)

if __name__ == '__main__':
    main()

# python -m torch.distributed.launch --nproc_per_node 1 --master_port 12345 main.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --data-path '/media/hello/453C61C6A3A805B6/AI/dataset/cat_or_dog_baidu/temp'  --batch-size 8 --local_rank 0 --zip  --accumulation-steps 2 --opts TRAIN.EPOCHS 20 TRAIN.BASE_LR 7.0 MODEL.NUM_CLASSES 12 SAVE_FREQ 5
# python infer.py --cfg configs/swin/swin_tiny_patch4_window7_224.yaml --weight output/swin_tiny_patch4_window7_224/default/ckpt_epoch_0.pth --data_path /media/hello/453C61C6A3A805B6/AI/dataset/cat_or_dog_baidu/cat_12_test/ --opts MODEL.NUM_CLASSES 12