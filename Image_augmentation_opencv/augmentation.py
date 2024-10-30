import random 
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt


class Augmentations:
    def __init__(self):
        # Dictionary ánh xạ tên augmentation thành hàm tương ứng
        self.augmentations = {
            'image_Fliped': self.image_Fliped,
            'image_Shifted': self.image_Shifted,
            'image_Rotation': self.image_Rotation,
            'image_Blur': self.image_Blur,
            'image_Resize': self.image_Resize,
            'image_Crop': self.image_Crop
        }

    def apply_augmentation(self, aug, image):
        # Gọi hàm từ dictionary bằng tên chuỗi `aug`
        if aug in self.augmentations:
            return self.augmentations[aug](image)
        else:
            print(f"Augmentation '{aug}' không hợp lệ.")
            return image

    def image_Fliped(self,image:np.ndarray,note = random.randint(-1,1))->np.ndarray:
        '''
        Flip the image horizontally or vertically;
        
        note = -1 flip along the x, y axis;
        note = 0 flip along the x axis ;
        note = 1 flip along the y axis ;

        input: image (np.ndarray)
        output: image is fliped
        '''
        return cv2.flip(image, note)
    def image_Shifted(self, image):
        '''
        Shift the image by a random x and y amount;
        
        input: image (np.ndarray)
        output: shifted image'''
        # 
        # image = cv2.resize(image,(640,480))
        rows, cols = image.shape[:2]    
        tx = random.randint(int(-.25*cols), int(.25*cols))
        ty = random.randint(int(-.25*rows), int(.25*rows))
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        shift_aug_img = cv2.warpAffine(image, M, (cols, rows))
        
        x, y = max(tx, 0), max(ty, 0)
        w, h = cols - abs(tx), rows - abs(ty)
        shift_aug_img = shift_aug_img[y:y+h, x:x+w] 
        shift_aug_img = cv2.resize(shift_aug_img, (cols, rows))
        return shift_aug_img
    
    def image_Rotation(self,image):
        '''
        Rotate the image by a random degree;
        
        input: image (np.ndarray)
        output: rotated image'''
        
        rows, cols = image.shape[:2] 
        Cx , Cy = rows, cols #center of rotation
        rand_angle = random.randint(-180,180) #random angle range
        M = cv2.getRotationMatrix2D((Cy//2, Cx//2),rand_angle ,1) #center angle scale
        aug_imgR = cv2.warpAffine(image, M, (cols, rows))  #apply rotation matrix such as previously explained
        return aug_imgR
    
    def image_Blur(self, image):
        '''
        Apply Gaussian blur to the image;
        
        input: image (np.ndarray)
        output: blurred image'''
        
        blur_val = random.randint(5,15) #blur value random
        aug_img = cv2.blur(image,(blur_val, blur_val))
        return aug_img
    
    def image_Resize(self, image: np.ndarray, interpolation=cv2.INTER_LINEAR) -> np.ndarray:
        '''
        Resize the image to a specified width and height;
        
        input: image (np.ndarray), 
        width (int) : the width of the image, 
        height (int) : the height of the image, 
        interpolation (int): Interpolation is determined
        
        output: resized image'''

        cols,rows = image.shape[:2]
        width = random.randint(int(0.8*cols), int(1.2*cols))
        height = random.randint(int(0.8*rows), int(1.2*rows))
        return cv2.resize(image, (width, height), interpolation=interpolation)
    
    def image_Crop(self,image: np.ndarray, crop_size: int = 100, num_crops :int=5) -> np.ndarray:
        '''
        Crop the image to a specified size;
        
        input: image (np.ndarray), 
        crop_size (int) : the size of the cropped image,
        num_crops (int) : the number of crops
        
        output: cropped image
        '''
        h, w = image.shape[:2]

        if crop_size > min(h, w):
            raise ValueError("crop_size is greater than image before cropping")

        masked_image = image.copy()

        for _ in range(num_crops):
            x_start = np.random.randint(0, w - crop_size + 1)
            y_start = np.random.randint(0, h - crop_size + 1)

            masked_image[y_start:y_start + crop_size, x_start:x_start + crop_size] = (0, 0, 0)

        return masked_image
