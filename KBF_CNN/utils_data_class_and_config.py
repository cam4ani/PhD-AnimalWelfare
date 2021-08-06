"""
Mask R-CNN
Copyright (c) 2018 Matterport, Inc.
Licensed under the MIT License (see LICENSE for details)
Written by Waleed Abdulla
"""
import os
import sys
import json
import datetime
import numpy as np
import skimage.draw
import pickle
import cv2
import glob

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import model as modellib, utils

# Path to trained weights file
#COCO_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Directory to save logs and model checkpoints, if not provided
# through the command line argument --logs
#DEFAULT_LOGS_DIR = os.path.join(ROOT_DIR, "logs")

data_type = 'KBF'

class DataConfig(Config):
    """Derives from the base Config class and overrides some values"""
    
    # Give the configuration a recognizable name
    NAME = "KBF_flickr"

    #Issue 708 form mask-rcnn: Following Keras convention, an epoch doesn't always mean a full pass through the dataset. Rather, the STEPS_PER_EPOCH config setting allows you to control the number of steps per epoch. You can use small epochs to get more frequent updates in TensorBoard, or you can set it such that it corresponds with a full pass through the dataset.
    #Therefore, images per epoch = STEPS_PER_EPOCH * IMAGES_PER_GPU * GPU_COUNT
    
    #multi-gpu on keras-maskrcnn isn't supported at the moment.
    GPU_COUNT = 1
    
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 2
    # Number of training steps per epoch (should be equal to sample size/batch size)
    STEPS_PER_EPOCH = int(30/(IMAGES_PER_GPU * GPU_COUNT))
    #batch size: 
    BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    
    # Number of validation steps to run at the end of every training epoch.
    VALIDATION_STEPS = int(5/(IMAGES_PER_GPU * GPU_COUNT))
    
    # Train or freeze batch normalization layers
    #     None: Train BN layers. This is the normal mode
    #     False: Freeze BN layers. Good when using a small batch size
    #     True: (don't use). Set layer in training mode even when inferencing
    TRAIN_BN = False #false as we have small batch size, otherwise might hurt perfo!  
    
    #avoid overfitting: resnet50 si non resnet101
    BACKBONE = "resnet50"

    #TOMODIFY
    #number of classes
    NUM_CLASSES = 1 + 4  # Background + type

    # Skip detections with < 50% confidence
    DETECTION_MIN_CONFIDENCE = 0.65 #CHANGED 0.65
    
    #maximum number of detected object: 8
    DETECTION_MAX_INSTANCES = 8
    
    # Loss weights for more precise optimization.
    # Can be used for R-CNN training setup.
    #rpn_class_loss : How well the Region Proposal Network separates background with objetcs
    #rpn_bbox_loss : How well the RPN localize objects
    #mrcnn_bbox_loss : How well the Mask RCNN localize objects
    #mrcnn_class_loss : How well the Mask RCNN recognize each class of object
    #mrcnn_mask_loss : How well the Mask RCNN segment objects'''
    #important
    LOSS_WEIGHTS = {
        "rpn_class_loss": 1., 
        "rpn_bbox_loss": 1.,
        "mrcnn_class_loss": 1., 
        "mrcnn_bbox_loss": 1.,
        "mrcnn_mask_loss": 1.
    }
        
    #Anchor stride
    #If 1 then anchors are created for each cell in the backbone feature map.
    #If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 2    
    #multiscale issue!!!  must be change if not good at detecting large object. region proposal network’s anchor box size
    # Length of square anchor side in pixels
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512) #TODOOOO: (16, 32, 64, 128, 256) (32, 48, 64, 128, 256)
    #TODO: anchor test and read (4, 8, 16, 32, 64)
    #If enabled, resizes instance masks to a smaller size to reduce memory load. Recommended when using high-resolution images.
    USE_MINI_MASK = False #initialy: True
    #MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
    
    #Image mean (RGB)
    MEAN_PIXEL = np.array([80, 80, 80])
    #MEAN_PIXEL = np.array([88.5, 86.1, 88.0]) #TODOOOOOOOOOOO was 123.7, 116.8, 103.9
    
    #Percent of positive ROIs used to train classifier/mask heads
    ROI_POSITIVE_RATIO = 0.33
    
    #How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    #ROIs kept after non-maximum supression (training and inference)
    POST_NMS_ROIS_TRAINING = 600 #TODO: was 2000 CHANGED 1000
    POST_NMS_ROIS_INFERENCE = 200 #TODO: was 1000 CHANGED 300
    #maximum number of ground truth instances to use in one image (reduce when images don't have a lot of objects)
    MAX_GT_INSTANCES = 30 #TODO: was 100
    #less if only one objet per classes
    #use fewer ROIs in training the second stage. This setting is like the batch size for the second stage of the model
    
    #ROI mini batch
    # Number of ROIs per image to feed to classifier/mask heads
    # The Mask RCNN paper uses 512 but often the RPN doesn't generate
    # enough positive proposals to fill this and keep a positive:negative
    # ratio of 1:3. You can increase the number of proposals by adjusting
    # the RPN NMS threshold.
    #higher: better for a lot of objects (one element might already product 20 ROI so if only 1 then 20:512 !=1:3)
    #higher better when 20 masks in one image. But lower: less OOM error
    TRAIN_ROIS_PER_IMAGE = 128
    # Non-maximum suppression threshold for detection
    # Non-max suppression threshold to filter RPN proposals.
    # You can increase this during training to generate more proposals.
    RPN_NMS_THRESHOLD = 0.2 #CHANGED 0.7 initial: 0.7 #TOTRY0: decrease to 0.4 to perhaps reduce overlaping output mask of same id
    DETECTION_NMS_THRESHOLD = 0.3 #initial: 0.3

    # Input image resizing
    # Generally, use the "square" resizing mode for training and inferencing
    # and it should work well in most cases. In this mode, images are scaled
    # up such that the small side is = IMAGE_MIN_DIM, but ensuring that the
    # scaling doesn't make the long side > IMAGE_MAX_DIM. Then the image is
    # padded with zeros to make it a square so multiple images can be put
    # in one batch.
    # Available resizing modes:
    # none:   No resizing or padding. Return the image unchanged.
    # square: Resize and pad with zeros to get a square image
    #         of size [max_dim, max_dim].
    # pad64:  Pads width and height with zeros to make them multiples of 64.
    #         If IMAGE_MIN_DIM is not None, then scale the small side to
    #         that size before padding. IMAGE_MAX_DIM is ignored in this mode.
    #         The multiple of 64 is needed to ensure smooth scaling of feature
    #         maps up and down the 6 levels of the FPN pyramid (2**6=64).
    IMAGE_RESIZE_MODE = "square" #"crop"
    IMAGE_MIN_DIM = 170 
    IMAGE_MAX_DIM = 256 
    
    # Learning rate, momentum and weight decay(regularisation)
    LEARNING_RATE = 0.0005     #not smaller than 0.001 0.001 worked!
    LEARNING_MOMENTUM = 0.9   
    WEIGHT_DECAY = 0.000001 #0.001
    

class VGG_Dataset(utils.Dataset):

    def load_vgg(self, dataset_dir, subset):
        """Load a subset of the data_type dataset.
        dataset_dir: Root directory of the dataset.
        subset: Subset to load: train or val
        """
        
        #to modify when having new species
        self.add_class(data_type, 1, "F") #source, id, class_name
        self.add_class(data_type, 2, "G") 
        self.add_class(data_type, 3, "GF")
        self.add_class(data_type, 4, "R")
        
        #Train or validation dataset?
        assert subset in ["train", "val"]

        #load annotations
        annotations = pickle.load(open(os.path.join(dataset_dir, 'annotation_'+subset+'.pkl'), 'rb'))
            
        #add images
        for a in annotations:
            self.add_image(
                data_type,
                image_id=a['filename'],  # use file name as a unique image id: '12232_C1.jpg'
                path=glob.glob(os.path.join(dataset_dir, a['filename'][0:-4]+'.*'))[0],
                width=a['width'], 
                height=a['height'],
                polygons=a['polygons'],
                class_id_=a['class_id_']
            ) #always add these new caracteristics in the annotations file
            
            
    def load_mask(self, image_id):
        """Generate instance masks for an image.
       outputs:
        masks: A bool array of shape [height, width, instance count] with one mask per instance.
        class_ids: a 1D array of class IDs of the instance masks.
        """
        # If not a data_type dataset image, delegate to parent class.
        image_info = self.image_info[image_id]
        if image_info["source"] != data_type:
            return super(self.__class__, self).load_mask(image_id)
        
        #load mask
        info = self.image_info[image_id]
        mask = np.zeros([info["height"], info["width"], len(info["polygons"])], dtype=np.uint8)
      
        for i, p in enumerate(info["polygons"]):
            #get indexes of pixels inside the polygon and set them to 1
            rr, cc = skimage.draw.polygon(p['all_points_y'], p['all_points_x'])
            mask[rr, cc, i] = 1
            
        #TO BE ADAPTED:
        #return mask, and array of class IDs of each instance.
        #if one class ID only, we return an array of 1s (of length 'nbr of masks', i.e. each mask represent an object)
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)
        #if more than one class id
        return mask.astype(np.bool), np.array(info['class_id_'])
        #return mask.astype(np.bool), np.ones([mask.shape[-1]], dtype=np.int32)

    
    def load_filename(self, image_id):
        """Load the specified image and return its filename.
        """
        return  self.image_info[image_id]['id']
    
    
    def image_reference(self, image_id):
        """Return the path of the image."""
        info = self.image_info[image_id]
        if info["source"] == data_type:
            return info["path"]
        else:
            super(self.__class__, self).image_reference(image_id)
            
   
