from torch.utils.data.dataset import Dataset
import os
import pandas as pd
import re
from PIL import Image

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

        # root / <label>  / <train/test> / <item> / <view>.png
        # Change here to read txt files directly

        data_type += '.txt'
        root = os.path.join(root,'sets')
        myset = os.path.join(root,data_type)
        print(myset)
        subjects = pd.read_csv(myset,header=None,sep=' ',names=['MVtxt','label'])
        #print(subjects.head(2),'\n')

        for idx, subject,label in subjects.itertuples():
            viewfiles = pd.read_csv(subject,header=None, sep=' ',names=['MVtxt','angle'],skiprows=2)
            views = []
            for i in range(len(viewfiles)):
                AA = re.search("Angle_[0-9]*",viewfiles.iloc[i,0]).group().upper()
                viewfiles.iloc[i,1] = int(AA[6:])
            viewfiles.sort_values(by='angle',inplace=True)
            views = viewfiles.MVtxt.tolist()
            views = views[::4]

            image_views = []

            for view in views:
                im = Image.open(view)
                im = im.convert('RGB')
                if self.transform is not None:
                    im = self.transform(im)
                image_views.append(im)

            #return views, self.y[index]

            self.x.append(image_views)
            self.y.append(label)
        #print(self.y)
        #print('Data Loaded!')

    # Override to give PyTorch access to any image on the dataset
    def __getitem__(self, index):
        orginal_views = self.x[index]
        views = []

        for im in orginal_views:
            #im = Image.open(view)
            #im = im.convert('RGB')
            if self.transform is not None:
                im = self.transform(im)
            views.append(im)

        return views, self.y[index]

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.x)
