from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import re
from PIL import Image
#import cv2
import sys
import tensorflow as tf
import numpy as np
#from augment import augmentImages

class MultiViewDataSet(Dataset):
    
    
    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes = ['Standard', 'AccB', 'AbsRB7']
        #classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, step = 1,transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform
        #self.sess = tf.Session(graph=g_1,config=tf.ConfigProto(log_device_placement=True,gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.02)))

        # root / <label>  / <train/test> / <item> / <view>.png
        # Change here to read txt files directly

        data_type += '.txt'
        root = os.path.join(root,'sets')
        myset = os.path.join(root,data_type)
        print(myset)
        subjects = pd.read_csv(myset,header=None,sep=' ',names=['MVtxt','label'])
        #subjects = subjects.iloc[:24,:]
        #print(subjects.head(2),'\n')
        c = 0
        Max = len(subjects)
        for idx, subject,label in subjects.itertuples():
            viewfiles = pd.read_csv(subject,header=None, sep=' ',names=['MVtxt','angle'],skiprows=2)
            views = []
            for i in range(len(viewfiles)):
                AA = re.search("Angle_[0-9]*",viewfiles.iloc[i,0]).group().upper()
                viewfiles.iloc[i,1] = int(AA[6:])
            viewfiles.sort_values(by='angle',inplace=True)
            views = viewfiles.MVtxt.tolist()
            views = views[::step]
            sys.stdout.flush()
            sys.stdout.write('Loading {} data: {:.0f}% \r'.format(data_type.split('.')[0],c*100/Max))

            c += 1
            #print((views))
            #halt
            image_views = []

            for view in views:
                im = Image.open(view)
                im = im.convert('RGB')
                if self.transform is not None:
                    im = self.transform(im)
                image_views.append(im)
            image_views = np.stack(image_views, axis=0)
            #return views, self.y[index]

            self.x.append(image_views)
            self.y.append(label)
            assert image_views.shape[0]==48,"Object with number of views other than 48 was found!"
        #print(self.x[0].shape)
        self.nofviews, _ , self.width , self.height = self.x[0].shape
        #print('Data Loaded!')

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        original_views = self.x[index]
        #views = []
        #print("shape of Orginal_Views: ", original_views.shape)
        #for im in original_views:
            #im = Image.open(view)
            #im = im.convert('RGB')
            #if self.transform is not None:
            #    im = self.transform(im)
            #views.append(im)
        #augment_on_GPU(original_views)
            #aug_views = augment_on_GPU(original_views)
        return original_views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
