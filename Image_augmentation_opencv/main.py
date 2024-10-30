#import 
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from  Image_augmentation_opencv.augmentation import Augmentations 

if __name__ == "__main__":

    augs_list = ['image_Fliped','image_Shifted','image_Rotation','image_Blur','image_Resize','image_Crop']

    Aug = Augmentations()
    
    for aug in augs_list:
        image = cv2.imread(r'E:\AI_DEV_LEARN\Image_augmentation_opencv\data\original\image.jpg')
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        
        image_aug = Aug.apply_augmentation(aug, image)
        
          # Tạo và lưu ảnh
        fig, ax = plt.subplots(ncols=2, sharex=True, figsize=(12, 12))
        ax[0].set_title("Original Image", fontsize=15)
        ax[0].imshow(image)
        ax[0].axis('off')
        
        ax[1].set_title(f"{aug} Image", fontsize=15)
        ax[1].imshow(image_aug)
        ax[1].axis('off')
        
        # Lưu ảnh với tên dựa trên augmentation
        plt.savefig(f"Image_augmentation_opencv/data/augmented/{aug}.png", format='png', dpi=300)
        plt.close(fig)  # Đóng hình để tránh hiển thị chồng chéo