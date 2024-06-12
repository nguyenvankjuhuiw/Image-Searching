# index.py
from extractfeatures import ExtractFeatures
import argparse
import glob
import cv2
import re
import csv

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True,
                help="Path to the directory that contains the images to be indexed")
ap.add_argument("-i", "--hog", required=True,
                help="Path to where the computed index will be stored")
args = vars(ap.parse_args())

# initialize the color descriptor
cd = ExtractFeatures()

# open the output index file for writing
with open(args["hog"], "w", newline='') as output:
    csv_writer = csv.writer(output)

    # use glob to grab the image paths and loop over them
    for imagePath in glob.glob(args["dataset"] + "/*.jpg"):
        # extract the image ID (i.e. the unique filename) from the image
        # path and load the image itself
        imageID = re.search(r'\d+', imagePath).group()
        image = cv2.imread(imagePath)

        if image is None:
            print("Không thể đọc hình ảnh")

        # describe the image
        features = cd.describe(image)

        # create a single row with all features for this image
        row = [imageID]
        for feature in features:
            row.append(" ".join(str(x) for x in feature))

        # write the row to the CSV file
        csv_writer.writerow(row)

print("Indexing completed!")


# python index.py --dataset dataset --index index.csv