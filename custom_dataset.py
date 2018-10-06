from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import re
from PIL import Image
#import cv2
import sys
import tensorflow as tf
import numpy as np
from augment import augmentImages

#GPU Augmentation Graph
'''
with tf.Graph().as_default():
        # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(None, 3, 224, 224), name='im0')
        # graph outputs
        with tf.device('/gpu:0'):
            view = tf.transpose(view_, perm=[0, 2, 3, 1])
            aug_view = augmentImages(view, 
                horizontal_flip=False, vertical_flip=False, translate = 64, rotate=30, crop_probability=0, mixup=0)
            aug_view = tf.transpose(aug_view, perm=[0, 3, 1 ,2])
        # build the summary operation based on the F colection of Summaries
        # must be after merge_all_summaries
        
        init_op = tf.global_variables_initializer()
        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.1))
        config.gpu_options.allow_growth = False
        sess = tf.Session(config=config)#config=tf.ConfigProto(log_device_placement=FLAGS.log_device_placement))
'''

# Creates a graph.
c = []
g_1 = tf.Graph()
with g_1.as_default():
#with tf.Graph().as_default():
    with tf.device('/cpu:0'):
                    # placeholders for graph input
        view_ = tf.placeholder('float32', shape=(48, 3, 224, 224), name='im0')
            
    for i in [3]:#range(4):
        with tf.device('/gpu:%d' % i):
            with tf.name_scope('%s_%d' % ('tower', i)) as scope:
            # graph outputs
                view = tf.transpose(view_[i::4], perm=[0, 2, 3, 1])
                aug_view = augmentImages(view, 
                    horizontal_flip=False, vertical_flip=False, translate = 64, rotate=30, crop_probability=0, mixup=0)
                c.append(tf.transpose(aug_view, perm=[0, 3, 1 ,2]))
                
            
    with tf.device('/cpu:0'):
      aug_view = tf.concat(c,axis=0)

# Creates a session with log_device_placement set to True.
sess = tf.Session(graph=g_1,config=tf.ConfigProto(log_device_placement=True,gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.4)))
# Runs the op.
##########
def augment_on_GPU(views):
    #list_of_augviews = []
    #print("Shape of views is:",views.shape)
        val_feed_dict = {view_: views}
        aug_views = sess.run(aug_view, feed_dict=val_feed_dict)
    #list_of_augviews.append(aug_views)
        #print("Shape of aug views is:",aug_views.shape)
            
    #inputs = np.stack(list_of_augviews, axis=1)
        return aug_views

class MultiViewDataSet(Dataset):
    
    
    def find_classes(self, dir):
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes = ['Standard', 'Abnormal']
        #classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        return classes, class_to_idx

    def __init__(self, root, data_type, transform=None, target_transform=None):
        self.x = []
        self.y = []
        self.root = root

        self.classes, self.class_to_idx = self.find_classes(root)

        self.transform = transform
        self.target_transform = target_transform
        self.sess = tf.Session(graph=g_1,config=tf.ConfigProto(log_device_placement=True,gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=.02)))

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
            #views = views[:36:1]
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
        #print(self.y)
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
        return augment_on_GPU(original_views), self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
