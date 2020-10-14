import os
import cv2
import numpy as np

""" 
To use this class:
initiate a class instance
siamese = SiameseNetworkDataset(data_folder, IMSIZE)
siamese.make_data('out_name')
The Data can be loaded from any other file by loading it through numpy:
np.load('out.npy', allow_pickle=True)
"""

class SiamesePairedDataset(object):
    """ 
    This class outputs a .npy file with the images and label as (img0,img1,label)
    it takes dataset path and image size as arguments
    """
    def __init__(self, PATH, IMG_SIZE):
        self.path = PATH
        self.img_size = IMG_SIZE
    
    LABELS = {'similar': 1, 'not_similar':0}
    
    training_data = []
    
    def num_data_items(self):
        i = 0
        file_names = []

        # First get all file names in list file_names
        for file in sorted(os.listdir(self.path)):
            file_names.append(str(file))
        return len(file_names)

    
    def make_data(self, out_name):
        # provide a name for .npy file being written
        i = 0
        file_names = []

        # First get all file names in list file_names
        for file in sorted(os.listdir(self.path)):
            file_names.append(str(file))

        for i in range(len(file_names)-2): 
            # Check if they are pairs
            if file_names[i][-5] == 'A' and file_names[i+1][-5] == 'B':
                img0_path = os.path.join(self.path, file_names[i])
                img1_path = os.path.join(self.path, file_names[i+1])

                # Read the image
                img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

                # Resize according to IMG_SIZE
                img0 = cv2.resize(img0, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img1 = cv2.resize(img1, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)

                # append to training_data
                self.training_data.append([np.array(img0), np.array(img1), np.eye(2)[self.LABELS['similar'] ]])

            # check if there's an A-C pair
            if file_names[i][-5] == 'A' and file_names[i+2][-5] == 'C':
                img0_path = os.path.join(self.path, file_names[i])
                img1_path = os.path.join(self.path, file_names[i+2])

                # Read the image
                img0 = cv2.imread(img0_path, cv2.IMREAD_GRAYSCALE)
                img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)

                # Resize according to IMG_SIZE
                img0 = cv2.resize(img0, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)
                img1 = cv2.resize(img1, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)

                # append to training_data
                self.training_data.append([np.array(img0), np.array(img1), np.eye(2)[self.LABELS['similar'] ]])

            # else, continue
            i += 1
        
        np.save(out_name + '.npy', self.training_data )

class SiameseDataset(object):
    """ 
    This class is meant to read and append all images to a npy file
    """
    def __init__(self, PATH, IMG_SIZE):
        self.path = PATH
        self.img_size = IMG_SIZE
    
    training_data = []

    def make_data(self, out_name):
        i = 0
        file_names = []

        # First get all file names in list file_names
        for file in sorted(os.listdir(self.path)):
            file_names.append(str(file))

        for i in range(len(file_names)):
            # get the image path
            img_path = os.path.join(self.path, file_names[i])

            # read image with cv
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # resize according to IMSIZE
            img = cv2.resize(img, (self.img_size, self.img_size), interpolation= cv2.INTER_CUBIC)

            # append to output list
            self.training_data.append(np.array(img))
        
        # continue incrementing i
        np.save(out_name + '.npy', self.training_data )

