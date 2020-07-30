import os
from PIL import Image, ImageFile
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as transforms 




vgg = nn.Sequential(
		    nn.Conv2d(3, 3, (1, 1)),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(3, 64, (3, 3)),
		    nn.ReLU(),  # relu1-1
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(64, 64, (3, 3)),
		    nn.ReLU(),  # relu1-2
		    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(64, 128, (3, 3)),
		    nn.ReLU(),  # relu2-1
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(128, 128, (3, 3)),
		    nn.ReLU(),  # relu2-2
		    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(128, 256, (3, 3)),
		    nn.ReLU(),  # relu3-1
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 256, (3, 3)),
		    nn.ReLU(),  # relu3-2
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 256, (3, 3)),
		    nn.ReLU(),  # relu3-3
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 256, (3, 3)),
		    nn.ReLU(),  # relu3-4
		    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(256, 512, (3, 3)),
		    nn.ReLU(),  # relu4-1, this is the last layer used
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 512, (3, 3)),
		    nn.ReLU(),  # relu4-2
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 512, (3, 3)),
		    nn.ReLU(),  # relu4-3
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 512, (3, 3)),
		    nn.ReLU(),  # relu4-4
		    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 512, (3, 3)),
		    nn.ReLU(),  # relu5-1
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 512, (3, 3)),
		    nn.ReLU(),  # relu5-2
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 512, (3, 3)),
		    nn.ReLU(),  # relu5-3
		    nn.ReflectionPad2d((1, 1, 1, 1)),
		    nn.Conv2d(512, 512, (3, 3)),
		    nn.ReLU()  # relu5-4
			)

vggnet = nn.Sequential(
			# encode 1-1
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 1-1
			# encode 2-1
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 2-1
			# encoder 3-1
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 3-1
			# encoder 4-1
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True), # relu 4-1
			# rest of vgg not used
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)			

class TrainSet(Dataset):
	def __init__(self, content_dir, style_dir, crop_size = 256):
		super().__init__()
		Image.MAX_IMAGE_PIXELS = None
		ImageFile.LOAD_TRUNCATED_IMAGES = True
		self.transform = transforms.Compose([
			transforms.Resize(512, interpolation=Image.BICUBIC),
			transforms.RandomCrop(crop_size),
			transforms.ToTensor(),
			transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
			])
		self.content_files = [content_dir+file_name for file_name in os.listdir(content_dir)]
		self.style_files = [style_dir+file_name for file_name in os.listdir(style_dir)]
	def __len__(self):
		return min(len(self.style_files), len(self.content_files))

	def __getitem__(self, index):
		content_img = Image.open(self.content_files[index]).convert('RGB')
		style_img = Image.open(self.style_files[index]).convert('RGB')
	
		content_sample = self.transform(content_img)
		style_sample = self.transform(style_img)

		return content_sample, style_sample











