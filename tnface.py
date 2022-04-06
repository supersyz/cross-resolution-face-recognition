import torch
import torch.nn as nn
from functools import partial
import os
import argparse
from timm.models.vision_transformer import VisionTransformer, _cfg
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_
import os
from torchvision import datasets, transforms
from PIL import Image
import numpy as np
from tqdm import tqdm

from scipy.io import loadmat, savemat
import matlab
import matlab.engine
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"


def get_transforms(mode, resize=256, grayed_prob=0.2, crop_size=224):
    import torchvision.transforms as t
    def subtract_mean(x):
        mean_vector = [91.4953, 103.8827, 131.0912]
        x *= 255.
        x[0] -= mean_vector[0]
        x[1] -= mean_vector[1]
        x[2] -= mean_vector[2]
        return x
    if mode == True:
        return t.Compose([
                    t.Resize(resize),
                    t.RandomGrayscale(p=grayed_prob),
                    t.RandomCrop(crop_size),
                    t.ToTensor(),
                    t.Lambda(lambda x: subtract_mean(x))
                ])
    else:
        return t.Compose([
                    t.Resize(resize),
                    t.CenterCrop(crop_size),
                    t.ToTensor(),
                    t.Lambda(lambda x: subtract_mean(x))
                ])

def generate_mat(models,data_path,mat_path,d,dis,input_size):

    # models=levit.LeViT_128S_Val()
    # models.cuda()
    #
    # static = torch.load(ckp)['model']
    # #static = static['model']
    # #print(type(static))
    # #print(static.keys())
    # models.load_state_dict(static)
    # models.eval()

    img2tensor = get_transforms(mode=False)


    path = data_path

    #生成probe
    mat1 = loadmat(os.path.join(path, 'probe_img_ID_pairs.mat'))['probe_set']
    listimg = []
    for i in range(len(mat1)):
        listimg.append(mat1[i][0][0])
    mat = np.zeros((len(listimg), 2048)).astype(np.float32)
    j = 0
    for i in tqdm(listimg):
        with torch.no_grad():
            # 加载数据集
            img = Image.open(os.path.join(path , 'Probe' , i))
            # transform_list = []
            # transform_list.append(transforms.Resize((256, 256), interpolation=3))
            # transform_list.append(transforms.CenterCrop(224))
            # transform_list.append(transforms.Resize((224, 224), interpolation=3))
            # transform_list.append(transforms.ToTensor())
            # transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            # img2tensor = transforms.Compose(transform_list)
            img = img2tensor(img).unsqueeze(0).cuda()
            x1 = models(img)
            x1 = x1[0].cpu().detach().numpy()
            mat[j] = x1
            j = j + 1
    savemat(mat_path+'probe.mat', {'probe_feature_map': mat})

    # 生成gallery
    mat1 = loadmat(os.path.join(path ,'gallery_match_img_ID_pairs.mat'))['gallery_set']
    listimg = []
    for i in range(len(mat1)):
        listimg.append(mat1[i][0][0])
    mat = np.zeros((len(listimg), 2048)).astype(np.float32)
    j = 0
    for i in tqdm(listimg):
        with torch.no_grad():
            # 加载数据集
            img = Image.open(os.path.join(path , 'Gallery_Match' , i))
            #transform_list = []
            # transform_list.append(transforms.Resize((256, 256), interpolation=3))
            # transform_list.append(transforms.CenterCrop(224))
            # transform_list.append(transforms.Resize((224, 224), interpolation=3))
            # transform_list.append(transforms.ToTensor())
            # transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
            # img2tensor = transforms.Compose(transform_list)
            img = img2tensor(img).unsqueeze(0).cuda()
            x1 = models(img)
            x1 = x1[0].cpu().detach().numpy()
            mat[j] = x1
            j = j + 1
    savemat(mat_path+'gallery.mat', {'gallery_feature_map': mat})
    if d:
        #生成distractor
        listimg = os.listdir(os.path.join(path,  'Gallery_Distractor'))
        listimg.sort(key=lambda listimg: (int('%04d' % int(listimg.split('_')[0]) + '%04d' % int(listimg.split('_')[1]) +
                                              '%04d' % int(listimg.split('_')[-3]) + '%04d' % int(listimg.split('_')[-2]) +
                                              '%04d' % int(listimg.split('_')[-1][4:-4]))))

        mat = np.zeros((len(listimg),2048)).astype(np.float32)
        j = 0
        for i in tqdm(listimg):
            with torch.no_grad():
                # 加载数据集
                img = Image.open(os.path.join(path ,'Gallery_Distractor' , i))
                # transform_list = []
                # # transform_list.append(transforms.Resize((500, 500), interpolation=3))
                # # transform_list.append(transforms.CenterCrop(224))
                # transform_list.append(transforms.Resize((224, 224), interpolation=3))
                # transform_list.append(transforms.ToTensor())
                # transform_list.append(transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)))
                # img2tensor = transforms.Compose(transform_list)
                img = img2tensor(img).unsqueeze(0).cuda()
                x1 = models(img)
                x1 = x1[0].cpu().detach().numpy()
                mat[j] = x1
                j = j + 1
        savemat(mat_path+'distractor.mat', {'distractor_feature_map': mat})

def got_accracy(mt_path):
    eng = matlab.engine.start_matlab()
    print('ok')
    a = eng.test_face_identification2(mt_path)
    #A = matlab.double([[1, 2], [5, 6]])
    #print(type(A), A.size, A)
    #print(a)
    eng.quit()
    return  a

def main(args):
    weights_path = os.listdir(args.weights_path)
    for weight_path in weights_path:
        args.weights = os.path.join(args.weights_path,weight_path)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    from utils import load_models
    assert args.weights
    models,tm = load_models('senet50_ft_pytorch.pth','cpu',args.weights)
    #models = levit.LeViT_128S_Val(distillation=args.dis)

    # pre_weights = torch.load(args.weights)['model']
    # pre_dict = {k: v for k, v in pre_weights.items() if models.state_dict()[k].numel() == v.numel()}
    # missing_keys, unexpected_keys = models.load_state_dict(pre_dict,strict=False)
    models.to(device)
    models.eval()
    savepath = args.weights.split('/')[-1][:-4]
   # savepath = args.weights[:-4]
    # savepath[2] = savepath[2][:-14]

    print(savepath)
    mtpath = savepath
    matpath = os.path.join(args.matpath,mtpath)
    # for param in models.features.parammeters():
    #     param.requires_grad = False
    generate_mat(models,args.datapath,matpath,args.d,args.dis,args.input_size)
    if args.g:
        return got_accracy(mtpath)

if __name__ == "__main__":
    # root = './VGG128S_soft_fz/'
    # ck = '18checkpoint.pth'
    # ckp = os.path.join(root,ck)
    # txt = os.path.join(root,ck.split('.')[0]+'.txt')
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda',type=str,default='0')
    parser.add_argument('--weights', type=str, default='../models_ckp_1_44097.pth')
    parser.add_argument('--weights_path', type=str, default='../models_ckp_1_44097.pth')
    parser.add_argument('--datapath', type=str, default=r'E:\datasets\tinyface\Testing_Set')
    parser.add_argument('--matpath', type=str, default='../feature')
    parser.add_argument('--d', type=int, default=0)
    parser.add_argument('--dis', type=int, default=None)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--g', type=int, default=0)
    args = parser.parse_args()
    main(args)
    # for i in range(30,31):
    # # for i in range(35,68):
    #     with open("./16_base_msceleb/16verifications.txt", "a") as f:
    #         f.write(str(i)+'epoch:')
    #         f.write('\n')

        #ckp = os.path.join(root,str(i)+'checkpoint.pth')
    #ckp = './base/30checkpoint.pth'
   # generate_mat(ckp)

