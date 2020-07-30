from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from utils import TrainSet
from AdaIN import StyleTransferNet
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--content_dir', type=str, default='./data/mscoco/', help='content data set path')
parser.add_argument('--style_dir', type=str, default='./data/wikiart', help='style data set path')
parser.add_argument('--epochs', type=int, default=1, help='training epoch number')
parser.add_argument('--resume', type=int, default=0, help='continues from epoch number')
parser.add_argument('--cuda', action='store_true', help='Using GPU to train')



def main():
	opt = parser.parse_args()
	check_points_dir = './results/check_points/'
	weights_dir = './results/weights/'
	train_set = TrainSet(opt.content_dir, opt.style_dir)
	batch_size = 8
	trainloader = DataLoader(dataset=train_set, num_workers =4, batch_size=batch_size, shuffle=True)
	vgg_model = torch.load('vgg_normalized.pth')
	net = StyleTransferNet(vgg_model)
	if torch.cuda.is_available() and opt.cuda:
		net.cuda()

	decoder_optimizer = optim.Adam(net.decoder.parameters(), lr=1e-6)
	running_loss = 0.0
	running_losses = []
	it = 0


	if opt.resume != 0:
		check_point = torch.load(check_points_dir + "check_point_epoch_" + str(opt.resume)+'.pth')
		net.load_state_dict(check_point['net'])
		decoder_optimizer.load_state_dict(check_point['decoder_optimizer'])
		it, running_losses = check_point['it'], check_point['running_losses']


	for epoch in range(1+opt.resume, opt.epochs+1):
		print("epoch: %i/%i" % (int(epoch), int(opt.epochs)))
		training_bar = tqdm(trainloader)
		training_bar.set_description('Running Loss: %f' % (running_loss))
		running_losses.append((it, running_loss))
		running_loss = 0
		  
		for content_sample, style_sample in training_bar:
			
			if torch.cuda.is_available() and opt.cuda:
				content_sample = content_sample.cuda()
				style_sample = style_sample.cuda()
			loss_content, loss_style = net([content_sample, style_sample])



			loss_tot = loss_content + 10 * loss_style
			loss_tot.backward()
			decoder_optimizer.step()
			running_loss += loss_tot.item() * style_sample.size(0)
			decoder_optimizer.zero_grad()
			sample_num += style_sample.size(0)
			if ((it) % 500 ==0) and it!= 0:
				
				running_loss /= sample_num
				print('')
				training_bar.set_description('Running Loss: %f' % (running_loss))
				
				running_losses.append((it, running_loss))
				
				running_loss = 0.0
				sample_num = 0  

			if ((it)% np.ceil(len(trainloader.dataset)/batch_size)== 0) and it!= 0:
				running_loss /= sample_num
				sample_num = 0
			it += 1

		check_point = {'decoder': net.decoder.state_dict(), 'decoder_optimizer': decoder_optimizer.state_dict(),
								'running_losses': running_losses, 'it': it}
		torch.save(check_point, check_points_dir+ 'check_point_epoch_%d.pth' % (epoch))
		torch.save(net.decoder.state_dict(), weights_dir+ 'decoder_epoch_%d.pth' % (epoch))	
		np.savetxt("running_losses", running_losses, fmt='%i,%f')						
if __name__ == '__main__':
	main()







