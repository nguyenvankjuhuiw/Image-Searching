# search.py
from extractfeatures import ExtractFeatures
from searcher import Searcher
import argparse
import cv2
import matplotlib.pyplot as plt 

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--index", required=True,
                help="Path to where the computed index is stored")
ap.add_argument("-q", "--query", required=True,
                help="Path to the query image")
ap.add_argument("-r", "--result-path", required=True,
                help="Path to the result path")
args = vars(ap.parse_args())

# initialize the image descriptor
cd = ExtractFeatures()

# load the query image and describe it
query = cv2.imread(args["query"])
query_resized = cv2.resize(query, (256, 256))
query_features = cd.describe(query_resized)

# print(len(query_features[5]))

# perform the search
searcher = Searcher(args["index"])
results = searcher.search(query_features)


query_features_hog=cd.describe_hog(query_resized)
print(len(query_features_hog))

# Hiển thị query image và các ảnh kết quả bằng matplotlib
fig, axes = plt.subplots(2, 3, figsize=(6, 4))

# Tắt các trục không sử dụng ở hàng trên cùng
axes[0, 0].axis('off')

# Hiển thị query image ở ô giữa của hàng trên cùng
axes[0, 1].imshow(cv2.cvtColor(query_resized, cv2.COLOR_BGR2RGB))
axes[0, 1].axis('off')
axes[0, 1].set_title('Ảnh tìm kiếm')

# Tắt các trục không sử dụng ở hàng trên cùng
axes[0, 2].axis('off')

# Hiển thị ba ảnh kết quả ở hàng dưới
for i, (score, resultID) in enumerate(results):
    result = cv2.imread(str(args["result_path"] + "/" + resultID + '.jpg'))
    axes[1, i].imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    axes[1, i].axis('off')
    axes[1, i].set_title(f'Kết quả {i+1} - {score:.2f}')

plt.tight_layout()
# plt.show()

# python search.py --index index.csv --query queries/3.jpg --result-path dataset