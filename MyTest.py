import os
import torch
import argparse
import numpy as np
import imageio
import sys
from tqdm import tqdm
from skimage import img_as_ubyte
import importlib
sys.path.append('')
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from models.dataset import test_dataset as EvalDataset



net_list = ['UEDGNet_iterative_pvt_antiartifact_laplace']
model_list = ['MyTrain']

ckp_path_list = []
for cur_model in model_list:
    ckp_path_list.append(cur_model + '/Net_epoch_best.pth')
    
def evaluator(model, val_root, map_save_path, trainsize=352):
    val_loader = EvalDataset(image_root=val_root + 'Imgs/',
                             gt_root=val_root + 'GT/',
                             testsize=trainsize)
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(val_loader.size)):
            image, gt, name, _ = val_loader.load_data()
            gt = np.asarray(gt, np.float32)

            image = image.cuda()

            output = model(image)
            output = F.interpolate(output[0][3], size=gt.shape, mode='bilinear', align_corners=False)
            output = output.sigmoid().data.cpu().numpy().squeeze()
            output = (output - output.min()) / (output.max() - output.min() + 1e-8)

            imageio.imsave(map_save_path + name, img_as_ubyte(output)) #change
            #print('>>> prediction save at: {}'.format(map_save_path + name))
            
for cur_net, cur_model in zip(net_list, model_list):
    globals()['UEDG'] = importlib.import_module('models.' + cur_net)
    txt_save_path = './exp_result/{}/'.format(cur_model)
    cur_ckp = './log/' + cur_model + '/Net_epoch_best.pth'


cnt = 0
for cur_net, cur_model in zip(net_list, model_list):
    cnt += 1
    print('{}/{}'.format(cnt, len(net_list)))
    
    cur_module = importlib.import_module('models.' + cur_net)
    txt_save_path = './exp_result/{}/'.format(cur_model)
    cur_ckp = './log/' + cur_model + '/Net_epoch_best.pth'
    os.makedirs(txt_save_path, exist_ok=True)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True

    model = cur_module.UEDGNet(channel=64, M=[8, 8, 8], N=[4, 8, 16])
    model.load_state_dict(torch.load(cur_ckp))
    model.eval()
    model.cuda() #change

    for data_name in ['CHAMELEON', 'CAMO', 'COD10K', 'NC4K']:
        map_save_path = txt_save_path + "/{}/".format(data_name)
        os.makedirs(map_save_path, exist_ok=True)
        evaluator(
            model=model,
            val_root='./dataset/TestDataset/' + data_name + '/',
            map_save_path=map_save_path,
            trainsize=352)
