import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

import numpy as np
from PIL import Image
import os
from constant import const
import gc

import collections
import pickle
import os.path
from pathlib import Path

from utils import *
from models import *

import warnings
warnings.filterwarnings("ignore")
torch.set_printoptions(precision=10)

os.environ['CUDA_DEVICES_ORDER'] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = const.GPU

dataset_name = const.DATASET
train_folder = const.TRAIN_FOLDER
test_folder = const.TEST_FOLDER
rr = const.REMOVE_RATIO

batch_size = 4
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


gen_net = genNet()
gen_net.to(device)

dis_net = disNet()
dis_net.to(device)

if os.path.isfile('Pytorch_Code/gen_net.chckpt'):
    gen_dict = torch.load("Pytorch_Code/gen_net.chckpt")
    gen_net.load_state_dict(gen_dict)
if os.path.isfile('Pytorch_Code/dis_net.chckpt'):
    dis_dict = torch.load("Pytorch_Code/dis_net.chckpt")
    dis_net.load_state_dict(dis_dict)

pil2tensor = transforms.ToTensor()
tensor2pil = transforms.ToPILImage()

i_loss_crit = nn.MSELoss() ##  Produces the same results experimentally as the custom defined intensity loss function. Both are L2 Losses.
g_optimizer = optim.Adam(gen_net.parameters(), lr = 1e-06)
d_optimizer = optim.Adam(dis_net.parameters(), lr = 1e-06)

data_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], ## As per paper, images are resized to 256x256 and normalized to [-1,1]
                             std=[0.5, 0.5, 0.5])
    ])

test_dataset = torchvision.datasets.ImageFolder(root=test_folder, transform=data_transform)

test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False
    )

g_optimizer.zero_grad()
d_optimizer.zero_grad()

num_vids = len(os.listdir(test_folder))
gts = []
gens = []

for i in range(num_vids):
    gts.append([]) 
    gens.append([])

for j in range(num_vids):
    for i in range(batch_size):
        gts[j].append(np.array(0))  ## First four images are 'past' frames so we can't gen/predict for them. Pad with 0s to keep array sizes nice
        gens[j].append(np.array(0))  


compute_matrices = True

my_file = Path("gts.pickle")
if my_file.is_file():
    print("Loading Ground Truth Images...")
    gts = pickle.load( open( "gts.pickle", "rb" ) )
    compute_matrices = False    
    print("GTs Loaded")

my_file = Path("gens.pickle")
if my_file.is_file():
    print("Loading Generated Images...")
    gens = pickle.load( open( "gens.pickle", "rb" ) )
    compute_matrices = False
    print("GENs Loaded")

if (compute_matrices):
    print("Processing Data and Computing Matrices...")
    count = 0
    q = []
    for batch in enumerate(test_loader):
        count += 1
        ## Process batch from data loader
        ## Their implementation seems (can't verify for sure) to use overlapping batches, ie: batch 1: [1,2,3,4], batch 2: [2,3,4,5], etc...
        ## The following processing code simulates that because I couldn't find a native way to do it in pytorch.
        idx = batch[0]
        imBatch = batch[1]       

        if len(q) < batch_size + 1:
            q += [imBatch]
            if len(q) < batch_size + 1:
                continue
        else:
            q[:-1] = q[1:]
            q[-1] = imBatch

        imBatch = [item[0] for item in q]
        vid_num = [item[1].numpy() for item in q]
        vid_num = torch.Tensor(np.asarray(vid_num).flatten())

        imTBatch = torch.Tensor(batch_size+1, 3, 256, 256)
        torch.cat(imBatch, out=imTBatch)

        imBatch = [imTBatch, vid_num]

        if(len(set(imBatch[1].numpy())) != 1):
            continue

        tp = []
        for k in range(0,batch_size):
            tp += [imBatch[0][k]]
        tp = tuple(tp)

        imgs = torch.cat(tp, dim=0).unsqueeze(0)

        ## Last image in the batch is our ground truth prediction frame
        gt = imBatch[0][batch_size].unsqueeze(0) 
        
        ## Transfer data to GPU
        imgs = imgs.to(device)
            
        ## Get Outputs from Generator
        output = gen_net(imgs).squeeze(0)

        if count % 100 == 0: #< 30:#
            out = tensor2pil(torch.cat((imBatch[0][-2], output.detach().cpu(), gt.detach().cpu().squeeze(0)), dim=2))
            out.show()
            #out.save(str(count)+".jpg")
            print(count)

        ## These operations take a lot of memory. If you run out, comment out the GTs and generate the pickles for the generations only, then re-run and generate for GTs only
        #gts[int(vid_num[0])].append(gt.detach().cpu().numpy())
        gens[int(vid_num[0])].append(output.unsqueeze(0).detach().cpu().numpy())

    print("Dumping gen data to pickle...")
    pickle.dump( gens, open( "gens.pickle", "wb" ) )
    print("Data dump complete.")


    ## The gens and gts computing/dumping takes up a lot of system memory, so I split the operations.
    for batch in enumerate(test_loader):
        count += 1
        ## Process batch from data loader
        ## Their implementation seems (can't verify for sure) to use overlapping batches, ie: batch 1: [1,2,3,4], batch 2: [2,3,4,5], etc...
        ## The following processing code simulates that because I couldn't find a native way to do it in pytorch.
        idx = batch[0]
        imBatch = batch[1]       

        if len(q) < batch_size + 1:
            q += [imBatch]
            if len(q) < batch_size + 1:
                continue
        else:
            q[:-1] = q[1:]
            q[-1] = imBatch

        imBatch = [item[0] for item in q]
        vid_num = [item[1].numpy() for item in q]
        vid_num = torch.Tensor(np.asarray(vid_num).flatten())

        imTBatch = torch.Tensor(batch_size+1, 3, 256, 256)
        torch.cat(imBatch, out=imTBatch)

        imBatch = [imTBatch, vid_num]

        if(len(set(imBatch[1].numpy())) != 1):
            continue

        ## Last image in the batch is our ground truth prediction frame
        gt = imBatch[0][batch_size].unsqueeze(0) 

        if count % 100 == 0:
            print(count)

        ## These operations take a lot of memory. If you run out, comment out the GTs and generate the pickles for the generations only, then re-run and generate for GTs only
        gts[int(vid_num[0])].append(gt.detach().cpu().numpy())

    print("Dumping gt data to pickle...")
    pickle.dump( gts, open( "gts.pickle", "wb" ) )
    print("Data dump complete.")

print("Computing PSNR values...")
my_file = Path("psnr.pickle")
if my_file.is_file():
    psnr_loss = pickle.load( open( "psnr.pickle", "rb" ) )
else:
    psnr_loss = psnr_list(gens, gts)
    pickle.dump( psnr_loss, open( "psnr.pickle", "wb" ) )

gt_loader = GroundTruthLoader()
gt_loaded = gt_loader(dataset=dataset_name, remove_ratio=4) #Remove ratio means 1 in N remains after being removal happens

print("Computing AUC values...")
auc = compute_auc(psnr_loss, gt_loaded)
print(dataset_name, " results -- AUC: ", auc)