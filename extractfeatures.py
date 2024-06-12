# colordescriptor.py
import numpy as np
import cv2
from skimage.feature import hog 
from skimage import feature
 
class ExtractFeatures:
    def __init__(self, bins=(8, 12, 3), orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2)):
        self.bins = bins
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block

    def describe(self, image):
        
        hsv_features = self.describe_hsv(image)

        
        hog_features = self.describe_hog(image)

       
        combined_features = hsv_features + hog_features
        return combined_features

    def describe_hsv(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []

        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            hist = self.histogram(image, cornerMask)
            features.append(hist)

        hist = self.histogram(image, ellipMask)
        features.append(hist)

        return features

    def describe_hog(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features = []

        (h, w) = gray.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]

        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(gray.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

        for (startX, endX, startY, endY) in segments:
            cornerMask = np.zeros(gray.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)

            hist = self.hog_features(gray, cornerMask)
            features.append(hist)

        hist = self.hog_features(gray, cornerMask)
        features.append(hist)

        # # Trích xuất đặc trưng HOG cho toàn bộ ảnh
        # hist = self.hog_features(gray, None)
        # features.append(hist)

        return features


    def histogram(self, image, mask):
        hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                            [0, 180, 0, 256, 0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def hog_features(self, image, mask):
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        features, _ = hog(masked_image, orientations=self.orientations,
                          pixels_per_cell=self.pixels_per_cell,
                          cells_per_block=self.cells_per_block,
                          visualize=True)
        return features
    def rgb_to_hsv(self, pixel):
        r , g, b = pixel
        r , g ,b = b / 255.0, g / 255.0, r / 255.0

        v = max(r,g,b)
        delta = v - min(r,g,b)

        if delta == 0:
            h = 0
            s = 0
        else:
            s = delta / v
            if r == v:
                h = (g - b) / delta
            elif g == v:
                h = 2 + (b - r) / delta
            else:
                h = 4 + (r - g) / delta
            h = (h / 6) % 1.0

        return [int(h*180), int(s*255), int(v*255)]

    def covert_image_rgb_to_hsv(self, img):
        hsv_image=[]
        for i in img:
            hsv_image2=[]
            for j in i:
                new_color=self.rgb_to_hsv(j)
                hsv_image2.append((new_color))
                hsv_image.append(hsv_image2)
            hsv_image=np.array(hsv_image)
        return hsv_image

    def my_calcHist(image, channels, histSize, ranges):
        # Khởi tạo histogram với tất cả giá trị bằng 0
        hist = np.zeros(histSize, dtype=np.int64)
        # Lặp qua tất cả các pixel trong ảnh
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                # Lấy giá trị của kênh màu được chỉ định
                bin_vals = [image[i, j, c] for c in channels]
                # Tính chỉ số của bin
                bin_idxs = [(bin_vals[c] - ranges[c][0]) * histSize[c] // (ranges[c][1] - ranges[c][0]) for c in range(len(channels))]
                # Tăng giá trị của bin tương ứng lên 1
                hist[tuple(bin_idxs)] += 1
        return hist
    
    def extract_rgb(self, img):
        data_RGB =[]
      
        # Đọc ảnh và chuyển đổi sang không gian màu HSV
        bins = [8, 8,8]
        ranges = [[0, 256], [0, 256], [0, 256]]
        # img_hsv=covert_image_rgb_to_hsv(img)
        hist_my = self.my_calcHist(img, [0, 1, 2], bins, ranges)
        embedding = hist_my.flatten()
        embedding[0]=0

    def extract_hsv(self, img):
        data_HSV=[]
       
        bins = [8,12,3]
        ranges = [[0, 180], [0, 256], [0, 256]]
        img_hsv=self.covert_image_rgb_to_hsv(img)
        hist_my = self.my_calcHist(img_hsv, [0, 1, 2], bins, ranges)
        # print(hist_my.shape)
        embedding = hist_my.flatten()
        embedding[0]=0
       
    def convert_image_rgb_to_gray(img_rgb, resize="no"):
        h, w, _ = img_rgb.shape
        # Create a new grayscale image with the same height and width as the RGB image
        img_gray = np.zeros((h, w), dtype=np.uint32)

        # Convert each pixel from RGB to grayscale using the formula Y = 0.299R + 0.587G + 0.114B
        for i in range(h):
            for j in range(w):
                r, g, b = img_rgb[i, j]
                gray_value = int(0.299*r + 0.587*g + 0.114*b)
                img_gray[i, j] = gray_value
        # print(gray_image.shape())
        if resize!="no":
            img_gray = cv2.resize(src=img_gray, dsize=(496, 496))
        return np.array(img_gray)
    def hog_feature(gray_img):# default gray_image
        # 1. Khai báo các tham số
        (hog_feats, hogImage) = feature.hog(gray_img, orientations=9, pixels_per_cell=(8 , 8),
            cells_per_block=(2,2), transform_sqrt=True, block_norm="L2",
            visualize=True)
        return hog_feats
    
    def extract_hog(self, img):
        data_hog=[]
     
        img_gray=self.convert_image_rgb_to_gray(img)
        embedding=self.hog_feature(img_gray)
        embedding = embedding.flatten()
       

                

