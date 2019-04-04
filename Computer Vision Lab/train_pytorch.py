import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import math
import random
import numpy as np
from PIL import Image
import os
from constant import const
from models import *
from utils import *

import collections

import warnings
warnings.filterwarnings("ignore")

## Args for FlowNet
if __name__ == '__main__': 
    parser = argparse.ArgumentParser()

    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--total_epochs', type=int, default=10000)
    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default = -1, help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default = [256, 256], help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0, help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default = 1.)

    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')

    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')

    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true', help='run inference (save flows to file) and every validation_frequency epoch')

    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default = [-1,-1], help='spatial size divisible by 64. default (-1,-1) - largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--save_flow', action='store_true', help='save predicted flows to file')

    parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")

    parser.add_argument('--skip_training', action='store_true')
    parser.add_argument('--skip_validation', action='store_true')

    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('--fp16_scale', type=float, default=1024., help='Loss scaling, positive power of 2 values can improve fp16 convergence.')

args = parser.parse_args([])

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER

batch_size = 4 #const.BATCH_SIZE #2
iterations = const.ITERATIONS
num_his = const.NUM_HIS
height, width = 256, 256
flow_height, flow_width = const.FLOW_HEIGHT, const.FLOW_WIDTH

l_num = const.L_NUM
alpha_num = const.ALPHA_NUM
lam_lp = const.LAM_LP
lam_gdl = const.LAM_GDL
lam_adv = const.LAM_ADV
lam_flow = const.LAM_FLOW
adversarial = (lam_adv != 0)

summary_dir = const.SUMMARY_DIR
snapshot_dir = const.SNAPSHOT_DIR

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

print("Initializing networks and dataloaders...")

gen_net = genNet()
gen_net.apply(init_weights)
gen_net.to(device)

dis_net = disNet()
dis_net.apply(init_weights)
dis_net.to(device)

flow_net = FlowNet2SD(args)
checkpoint = torch.load('FlowNet2-SD_checkpoint.pth.tar')
flow_net.load_state_dict(checkpoint["state_dict"])
flow_net.to(device)

if os.path.isfile('gen_net.chckpt'):
    gen_dict = torch.load("gen_net.chckpt")
    gen_net.load_state_dict(gen_dict)
if os.path.isfile('dis_net.chckpt'):
    dis_dict = torch.load("dis_net.chckpt")
    dis_net.load_state_dict(dis_dict)

################################################################
################################################################
################################################################

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()


lrate = 1e-05
i_loss_crit = nn.MSELoss() ##  Produces the same results experimentally as the custom defined intensity loss function. Both are L2 Losses.
g_optimizer = optim.Adam(gen_net.parameters(), lr = lrate) 
d_optimizer = optim.Adam(dis_net.parameters(), lr = lrate)

data_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], ## As per paper, images are resized to 256x256 and normalized to [-1,1]
                             std=[0.5, 0.5, 0.5])
    ])

train_dataset = torchvision.datasets.ImageFolder(root=train_folder, transform=data_transform)

datasets = []
loaders = []
iterators = []
num_dir = os.listdir(train_folder)
for i in num_dir:
    ds = torchvision.datasets.ImageFolder(root=train_folder+"/"+i+"/", transform=data_transform)
    datasets += [ds]
    loader = torch.utils.data.DataLoader(
                ds,
                batch_size=batch_size+1,
                num_workers=0,
                drop_last=True,
                shuffle=False)
    loaders += [loader]
    #iterators += [iter(loader)]

g_optimizer.zero_grad()
d_optimizer.zero_grad()

print("Beginning Training...")
max_psnr = 0.0
for epoch in range(0, 40): ## 40+ Epochs for full, 100+ for 1-in-4
    gts = []
    gens = []
    idx = 0

    batches = []
    iterators = []
    num_dir = os.listdir(train_folder)
    for load in loaders:
        iterators += [iter(load)]

    
    nlr = lrate * (0.1 ** math.floor(epoch/15)) #15+ for full, 40+ for 1-in-4
    for param_group in g_optimizer.param_groups:
        param_group['lr'] = nlr
    for param_group in d_optimizer.param_groups:
        param_group['lr'] = nlr

    for b1 in enumerate(loaders[0]):
        batches = [b1[1]]
        for j in range(1,len(iterators)):
            try:
                batches += [next(iterators[j])]
            except StopIteration:
                pass

        random.shuffle(batches)


        for imBatch in batches:#enumerate(train_loader):
            ## Process batch from data loader
            ## Their implementation seems (can't verify for sure) to use overlapping batches, ie: batch 1: [1,2,3,4], batch 2: [2,3,4,5], etc...
            ## The following processing code simulates that because I couldn't find a native way to do it in pytorch without writing my own custom dataloaders
            ## and at this point, doing this hack was much faster/easier

            idx = idx + 1
            #imBatch = batch

            #imBatch = [imTBatch, vid_num]

            #if(len(set(imBatch[1].numpy())) != 1):
                #print(len(set(imBatch[1].numpy())))
            #    continue

            tp = []
            for k in range(0,batch_size):
                tp += [imBatch[0][k]]
            tp = tuple(tp)

            imgs = torch.cat(tp, dim=0).unsqueeze(0)

            ## Last image in the batch is our ground truth prediction frame
            gt = imBatch[0][batch_size].unsqueeze(0)

            ## Transfer data to GPU
            gt, imgs = gt.to(device), imgs.to(device)
            
            ## Get Outputs from Generator
            output = gen_net(imgs).squeeze(0)

            ## Intensity Loss
            int_loss = i_loss_crit(output, gt)

            ## Gradient Loss
            grad_loss, grad_out_im = compute_grad_loss(gen_frame=output, gt_frame=gt, alpha=alpha_num)

            ## Flow Loss
            ## Resize as per comments in the paper's tensorflow code
            inr1 = (F.upsample(imBatch[0][batch_size-1].to(device).unsqueeze(0), size=(384,512), mode='bilinear') + 1.0) / 2.0
            inr2 = (F.upsample(imBatch[0][batch_size].to(device).unsqueeze(0), size=(384,512), mode='bilinear') + 1.0) / 2.0
            inputs_r = torch.cat((inr1, inr2), dim=0).to(device)

            inf1 = (F.upsample(imBatch[0][batch_size-1].to(device).unsqueeze(0), size=(384,512), mode='bilinear') + 1.0) / 2.0
            inf2 = (F.upsample(output.to(device).unsqueeze(0), size=(384,512), mode='bilinear') + 1.0) / 2.0
            inputs_f = torch.cat((inf1, inf2), dim=0).to(device)            

            inputs_r = inputs_r.permute(1, 0, 2, 3).unsqueeze(0)
            inputs_f = inputs_f.permute(1, 0, 2, 3).unsqueeze(0)

            flow_out_real = flow_net(inputs_r).squeeze()
            flow_out_fake = flow_net(inputs_f).squeeze()
            
            flow_loss = torch.mean(torch.abs(flow_out_fake - flow_out_real)) * 0.005
            

            ## Discriminator Classifications of Fake/Real images
            f_logits, f_predictions = dis_net(output.unsqueeze(0))
            r_logits, r_predictions = dis_net(gt)

            ## Generative and Discriminative loss from GAN
            gen_loss = torch.mean((f_predictions-1).pow(2)/2) 
            dis_loss = torch.mean((r_predictions-1).pow(2)/2) + torch.mean((f_predictions).pow(2)/2)

            ## Total Generative and Discriminative loss
            g_loss = int_loss + grad_loss + flow_loss + 0.05 * gen_loss ## 0.05 times adversarial loss, as per paper
            #d_loss = dis_loss


            gts += [gt.detach().cpu()]
            gens += [output.unsqueeze(0).detach().cpu()]

            ## Freeze discriminator parameters
            #for param in dis_net.parameters():
            #    param.requires_grad = False
            #for param in gen_net.parameters():
            #    param.requires_grad = True

            ## Optimize Generator
            g_loss.backward(retain_graph=True)
            g_optimizer.step()
            g_optimizer.zero_grad()

            ## Unfreeze Discrim, Freeze Gen
            #for param in dis_net.parameters():
            #    param.requires_grad = True
            #for param in gen_net.parameters():
            #    param.requires_grad = False

            d_output = gen_net(imgs).squeeze(0)
            d_f_logits, d_f_predictions = dis_net(d_output.unsqueeze(0))
            d_r_logits, d_r_predictions = dis_net(gt)
            d_loss = torch.mean((d_r_predictions-1).pow(2)/2) + torch.mean((d_f_predictions).pow(2)/2)

            ## Optimize Discriminator
            d_loss.backward()
            d_optimizer.step()
            d_optimizer.zero_grad()

            ## Output for debugging, shows generated image vs real image on top, and the corresponding gradient images on bottom
            if idx % 50 == 0:
                print("Epoch ", epoch, "Iteration: ", idx, "Intensity Loss: ", int_loss.detach().cpu().numpy(), "Gradient Loss: ", 
                        grad_loss.detach().cpu().numpy(), "Flow Loss: ", flow_loss.detach().cpu().numpy(), "Gen-Loss: ", 0.05 * gen_loss.detach().cpu().numpy(), "Disc-Loss: ", d_loss.detach().cpu().numpy())  
            if idx % 200 == 0:
                im = torch.cat((torch.cat((output.detach().cpu(), gt.detach().cpu().squeeze(0)), dim=2), grad_out_im), dim=1)
                out = tensor2pil(im)
                out.show()

    gts = tuple(gts)
    gts = torch.cat(gts, dim=0)
    gens = tuple(gens)
    gens = torch.cat(gens, dim=0)
    psnr_loss = psnr(gens, gts)
    print("Epoch: ", epoch, "Iteration: ", idx, ", PSNR: ", psnr_loss)
                
    gts = []
    gens = []
    
    if (psnr_loss > max_psnr):
        max_psnr = psnr_loss
        print("PSNR Improvement, Saving Checkpoint...")

        torch.save(gen_net.state_dict(), "gen_net.chckpt")
        torch.save(dis_net.state_dict(), "dis_net.chckpt")
    elif(max_psnr == 0.0):
        max_psnr = psnr_loss
    
 