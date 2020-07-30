import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



mse_criterion = nn.MSELoss()
def calc_mean_std(feat, eps=1e-5):
	size = feat.size()
	assert(len(size)==4)
	N, C = size[:2]
	feat_var = feat.view(N, C, -1).var(dim=2) +eps
	feat_std = feat_var.sqrt().view(N, C, 1, 1)
	feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
	return feat_mean, feat_std

def calc_content_loss(input_im, target):
	assert (input_im.size() == target.size())
	assert (target.requires_grad is False)
	return mse_criterion(input_im, target)

def calc_style_loss(input_im, target):
	assert (input_im.size() == target.size())
	assert (target.requires_grad is False)
	input_mean, input_std = calc_mean_std(input_im)
	target_mean, target_std = calc_mean_std(target)
	return mse_criterion(input_mean, target_mean) + \
			mse_criterion(input_std, target_std)


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


class AdaIN(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, x, y):
		eps = 1e-5	
		mean_x = torch.mean(x, dim=[2,3])
		mean_y = torch.mean(y, dim=[2,3])

		std_x = torch.std(x, dim=[2,3])
		std_y = torch.std(y, dim=[2,3])

		mean_x = mean_x.unsqueeze(-1).unsqueeze(-1)
		mean_y = mean_y.unsqueeze(-1).unsqueeze(-1)

		std_x = std_x.unsqueeze(-1).unsqueeze(-1) + eps
		std_y = std_y.unsqueeze(-1).unsqueeze(-1) + eps

		out = (x - mean_x)/ std_x * std_y + mean_y


		return out

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std



class StyleTransferNet(nn.Module):
	def __init__(self, vgg_model):
		super().__init__()

		vggnet.load_state_dict(vgg_model)

		self.encoder = nn.Sequential(
			nn.Conv2d(3, 3, kernel_size=(1,1), stride= (1, 1)),
			nn.Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),

			nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),


			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False),
			
			nn.Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True)
			)

		self.encoder.load_state_dict(vggnet[:21].state_dict())
		for parameter in self.encoder.parameters():
			parameter.requires_grad = False
			
		self.decoder = nn.Sequential(
			nn.Conv2d(512, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(256, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(128, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),

			nn.Upsample(scale_factor=2, mode='nearest'),
			nn.Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),
			nn.ReLU(inplace=True),
			nn.Conv2d(64, 3, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), padding_mode='reflect'),

			)
		self.adaIN = AdaIN()
		self.mse_criterion = nn.MSELoss()




	def forward(self, x, alpha=1.0):

		content_img = x[0]
		style_img = x[1]

		encode_content = self.encoder(content_img)
		encode_style = self.encoder(style_img)

		encode_out = self.adaIN(encode_content, encode_style)
		

		if self.training:
			gen_img = self.decoder(encode_out)
			encode_gen = self.encoder(gen_img)

			fm11_style = self.encoder[:3](style_img)
			fm11_gen = self.encoder[:3](gen_img)

			fm21_style = self.encoder[3:8](fm11_style)
			fm21_gen = self.encoder[3:8](fm11_gen)

			fm31_style = self.encoder[8:13](fm21_style)
			fm31_gen = self.encoder[8:13](fm21_gen)
			
			loss_content = self.mse_criterion(encode_gen, encode_out)

			loss_style = self.mse_criterion(torch.mean(fm11_gen, dim=[2,3]), torch.mean(fm11_style, dim=[2,3])) +	\
						self.mse_criterion(torch.mean(fm21_gen, dim=[2,3]), torch.mean(fm21_style, dim=[2,3])) +	\
						self.mse_criterion(torch.mean(fm31_gen, dim=[2,3]), torch.mean(fm31_style, dim=[2,3])) +	\
						self.mse_criterion(torch.mean(encode_gen, dim=[2,3]), torch.mean(encode_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(fm11_gen, dim=[2,3]), torch.std(fm11_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(fm21_gen, dim=[2,3]), torch.std(fm21_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(fm31_gen, dim=[2,3]), torch.std(fm31_style, dim=[2,3])) +	\
						self.mse_criterion(torch.std(encode_gen, dim=[2,3]), torch.std(encode_style, dim=[2,3])) 

			return loss_content, loss_style
		encode_out = alpha * encode_out + (1-alpha)* encode_content 
		gen_img = self.decoder(encode_out)
		return gen_img























