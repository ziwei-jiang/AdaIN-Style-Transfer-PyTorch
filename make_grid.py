import argparse
import os, glob
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np



def resize_and_fill(img, size):
	width, height = img.size
	max_size = max(width, height)
	new_img = Image.new('RGB', (size, size))
	if width >= height:
		img = img.resize((size, int(height/width*size)), resample=Image.BICUBIC)
	else:
		img = img.resize((int(width/height*size), size), resample=Image.BICUBIC)
	width, height = img.size
	max_size = max(width, height)
	new_img.paste(img, (int((max_size-width)/2), int((max_size-height)/2)))
	return new_img


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default='./', help='The directory for the images')
parser.add_argument('--r', type=int, default=1, help='Numer of rows')
parser.add_argument('--c', type=int, default=1, help='Numer of columns')
parser.add_argument('--size', type=int, default=256, help='Size per thumbnail')
opt =parser.parse_args()

## Get all the png and jpg file in the target directory

files = glob.glob(opt.dir+'/*.png')
files.extend(glob.glob(opt.dir+'/*.jpg'))
files = sorted(files)

bg = Image.new('RGB', (opt.c*(opt.size+10)+10, opt.r*(opt.size+10)+10), (255, 255, 255))


## Choose images row by row
for r in range(opt.r):
	print("")

	for n,f in enumerate(files,1):
		print(n,f)

	## Get the image index from user input
	while True:
		try:
			row_names = input("Please input %d index for the row number %d, separate by space\n" % (opt.c,r+1)).split()
			row_idxs = [int(row_name)-1 for row_name in row_names]
		except ValueError:
			print('Input the index of image only')
			continue
		if len(row_idxs) != opt.c:
			print('Choose %d images for each row' % opt.c)
			continue
		else:
			break

	for c, c_idx in enumerate(row_idxs):
		if c_idx>=0:
			img = Image.open(files[c_idx])
			img_resize = resize_and_fill(img, opt.size)
			bg.paste(img_resize, (int(10 + c*(opt.size+10)) ,int(10 + r*(opt.size+10))))

		else:
			continue


bg.save('out.png')






