import torch
from segment_anything import sam_model_registry
import argparse
import cv2
from segment_anything import SamAutomaticMaskGenerator
from PIL import Image
import torchvision.transforms as transforms
import os
from matplotlib import pyplot as plt

#pip install 'git+https://github.com/facebookresearch/segment-anything.git'
#wget -q 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth'

parser = argparse.ArgumentParser(description='Get Segmentation Mask', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_type', type=str, default="vit_h", dest='model_type')
parser.add_argument('--input_image', type=str, default='./dataset/loopy/rgb/001.png', dest='input_path')
parser.add_argument('--output_folder', type=str, default='./dataset/loopy', dest='output_folder')
parser.add_argument('--device', type=str, default='cpu', dest='device')

args = parser.parse_args()
MODEL_TYPE = args.model_type
INPUT_IMAGE_PATH = args.input_path
OUTPUT_IMAGE_FOLDER = args.output_folder
DEVICE = args.device if torch.cuda.is_available() else 'cpu'


sam = sam_model_registry[MODEL_TYPE](checkpoint='./sam_vit_h_4b8939.pth')
sam.to(device=DEVICE)

mask_generator = SamAutomaticMaskGenerator(sam)

image_bgr = cv2.imread(INPUT_IMAGE_PATH)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
result = mask_generator.generate(image_rgb)

num_columns = 5
num_rows = (len(result) // num_columns) + 1
fig, ax = plt.subplots(num_rows, num_columns, figsize=(30,6*num_rows))

for i, ax in enumerate(ax.flatten()):
        if i < len(result):
            ax.imshow(result[i]['segmentation'], cmap='gray')
            ax.axis('off')
            ax.set_title(i+1)

plt.show()

select = input('type in number of segmentation map to choose: ')
try:
    select = int(select)
except:
    raise TypeError('only integer type of input is allowed')

if select == 0 or select > len(result):
    raise ValueError(f'only integer between 0, {len(result)} is allowed')

transform = transforms.Compose([
    transforms.Resize(512),
])

print(f'saving {select}th segmentation mask')
image = Image.fromarray(result[select-1]['segmentation'])
image = transform(image)
image.save(os.path.join(OUTPUT_IMAGE_FOLDER,f'mask_obj.png'), format='PNG')