import cv2
import numpy as np
import os
import random
import csv
from matplotlib import pyplot as plt
import glob
# train_path='../complete_ms_data/train'
# test_path='../complete_ms_data/test'

def get_impath(train_path='/home/amax/project/hs_re/complete_ms_data',test_path='../complete_ms_data/test'):
    flsit = os.listdir(train_path)
    image_list = []
    for f in flsit:
        f1=os.listdir(os.path.join(train_path,f))[0]
        images_path=os.path.join(train_path,f,f1)
        names = os.listdir(images_path)
        names.sort()
        listT=[]
        for name in names[2:]:
            image_path=os.path.join(train_path,f,f1,name)
            listT.append(image_path)
        image_list.append(listT)
    return image_list
def get_impath_nt20(train_path='/home/amax/project/hs_re/AWAN-master/AWAN_Clean/train/Dataset/',train=True,mode=None):
    if train:
        flsit =['Train1','Train2','Train3','Train4']
    elif mode=='test':
        flsit = ['test']
    else:
        flsit=['Valid']
    image_list = []

    for f in flsit:

        images_paths=os.path.join(train_path,f,'*mat')
        names = glob.glob(images_paths)
        names.sort()
        #listT=[]
        for name in names:
            image_path=name
            #listT.append(image_path)
            image_list.append(image_path)
    return image_list

def get_impath_RGB(train_path='/home/amax/project/hs_re/complete_ms_data',test_path='../complete_ms_data/test'):
    flsit = os.listdir(train_path)
    image_list = []
    for f in flsit:

        images_path=os.path.join(train_path,f,f)
        names = os.listdir(images_path)
        names.sort()
        listT=[]
        for name in names[1:2]:
            image_path=os.path.join(train_path,f,f,name)
            listT.append(image_path)
        image_list.append(listT)
    print(image_list)
    return image_list

def load_sr(path):
    #path = '/media/ylt/data/hs_re/srdata/%d.csv'%path
    with open(path,"r") as f :
        reader=csv.reader(f)
        dis={}
        for i,row in enumerate(reader):
            if i<2:
                continue
            try:

                di={int(row[0]):float(row[1])}
            except:
                print(row[1])
            dis.update(di)
    return dis


def load_all_sr():
    all_mat=np.zeros([1020,31,1,1]).astype(np.float32)
    wavelen_start = 400
    wavestep = 10
    slist=os.listdir('/data/ylt/project/hs_re/srdata')
    slist.sort()
    l=[454,
    262,
    842,
    609,
    943,
        ]
    for i in range(5):
     print(slist[l[i]])

    for f,sr in enumerate(slist):
        sr=os.path.join('/data/ylt/project/hs_re/srdata',sr)
        sr_dic=load_sr(sr)
        for i in range(31):
            #print(wavelen_start + i * wavestep)
            try:
                we = sr_dic[wavelen_start + i * wavestep]
            except:
                we = 0
            all_mat[f,i,0,0]=we/100
    return all_mat



def load_camera(ind):
    srlist = []
    outlist = []
    with open('/home/amax/project/hs_re/AWAN-master/AWAN_Clean/train/camspec_database.txt', 'r') as f:

        all_mat = np.zeros([3, 31, 1, 1]).astype(np.float32)
        for i, line in enumerate(f.readlines()):
            l = [4 * t for t in range(32)]
            l.append(117)
            if i not in l:
                srlist.append(np.array(line.split('\t')).astype(float))
    print(len(srlist))
    for i in range(0, len(srlist), 3):
        # print(1)

        o = np.stack([srlist[i], srlist[i + 1], srlist[i + 2]], axis=0)
        all_mat[:, :, 0, 0] = o[:, 0:31]
        outlist.append(all_mat.copy())
    return outlist[ind]




def conv(sr_dic,im):
    wavelen_start=400
    wavestep=10
    im_return =np.zeros_like(im[:,:,0]).astype(np.int32)
    for i in range(im.shape[2]):
        try:
            we=sr_dic[wavelen_start + i * wavestep]
        except:
            we=0
        i_t=im[:,:,i] *we

        im_return+=i_t.astype(np.int32)
        im_return=im_return / im.max() * 255
    return im_return


def show_im():
    image_path = get_impath()[1]
    print(image_path[0])
    print(image_path[0])
    im_stack = np.reshape(cv2.imread(image_path[0], -1),
                          [cv2.imread(image_path[0], -1).shape[0], cv2.imread(image_path[0], -1).shape[1], 1])
    print(im_stack.shape)
    for im_channel in image_path[1:]:
        image = np.reshape(cv2.imread(im_channel, -1),
                           [cv2.imread(im_channel, -1).shape[0], cv2.imread(im_channel, -1).shape[1], 1])
        im_stack = np.concatenate((im_stack, image), axis=2)
    n = random.randint(0, 256)
    m = random.randint(0, 256)
    im_corp = im_stack[ n:n + 256, m:m + 256,:]
    sr1=load_sr(2802)
    sr2=load_sr(2276)
    sr3 = load_sr(3969)
    c1= conv(sr1,im_corp)
    c2 = conv(sr2, im_corp)
    c3 = conv(sr3, im_corp)
    im_f=np.stack((c1,c2,c3),axis=2)
    print(im_f.shape)

    cv2.namedWindow('1')
    cv2.imshow('1',im_f)
    cv2.waitKey(10000000)
def plot_sr():
    sr1 = load_sr(2802)
    sr2 = load_sr(2276)
    sr3 = load_sr(3969)
    sr4=load_sr(3288)
    sr5=load_sr(4159)
    #sr2=sr2[:500]
    srlist=[sr1,sr2,sr3]#,sr4,sr5]
    c=['r','g','b']
    for i ,sr in enumerate(srlist):
        x=list(sr.keys())
        print(x)
        xx=[]
        y=[]

        for key in x:
            if float(key)>800:
                continue
            xx.append(key)
            y.append(sr[key])
        print(y)
        #plt.figure()
        plt.plot(xx,y,c[i])
    plt.savefig('/home/amax/project/hs_re/Image_Segmentation-master/img/sr.jpg')



if __name__=="__main__":
    if 0:
        print(get_impath())
        image_path = get_impath()[5]
        print(image_path[0])
        im_stack =np.reshape( cv2.imread(image_path[0],-1),[cv2.imread(image_path[0],-1).shape[0],cv2.imread(image_path[0],-1).shape[1],1])
        print(im_stack.shape)
        for im_channel in image_path[1:]:
            image = np.reshape( cv2.imread(im_channel,-1),[cv2.imread(im_channel,-1).shape[0],cv2.imread(im_channel,-1).shape[1],1])
            im_stack = np.concatenate((im_stack, image),axis=2)
        n = random.randint(0, 256)
        m = random.randint(0, 256)
        im_corp = im_stack[n:n + 256, m:m + 256, :]
        print(n,m,im_corp.max())
    else:
        # path='/media/ylt/data/hs_re/srdata/1842.csv'
        # di=load_sr(path)
        # print(di)
        #plot_sr()
        #get_impath_nt20(train=False)
        a=np.squeeze(load_all_sr())
        b=np.squeeze(load_camera(0))
        #get_impath_RGB()
