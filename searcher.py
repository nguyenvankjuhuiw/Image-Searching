# searcher.py
import numpy as np
import csv
import sys
csv.field_size_limit(2147483647)

class Searcher:
    def __init__(self, indexPath):
        self.indexPath = indexPath

    def search(self, queryFeatures, limit=3):
        results = {}
        with open(self.indexPath) as f:
            reader = csv.reader(f)
            kt=0
            for row in reader:
                imageID = row[0]
                # features = [list(map(float, x.split(','))) for x in row[1:]]
                
                features = []
                # print(len(row))
                # print(len(row))
                for i in range(1, len(row)):
                    features.append(float(i) for i in row[i].split(' '))
                # l=[]
                # for i in features:
                #     l.append(list(i))
                # if kt==0:
                #     print(l)
                #     kt=1
                d = self.weighted_distance(features, queryFeatures)
                results[imageID] = d
        f.close()
        results = sorted([(v, k) for (k, v) in results.items()])
        return results[:limit]

    def chi2_distance(self, histA, histB, eps=1e-10):
        d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
                          for (a, b) in zip(histA, histB)])
        return d
    
    # def distance_euclidean(self, x, y):
    #     x = list(x)  # Chuyển đổi generator thành danh sách
    #     y = list(y)  # Chuyển đổi generator thành danh sách
    #     if len(x) != len(y):
    #         return None 
    #     squared_distance = 0
    #     for i in range(len(x)):
    #         squared_distance += (x[i] - y[i]) ** 2
    #     return squared_distance ** 0.5 

    def weighted_distance(self, featuresA, featuresB):
        weights = [0.05, 0.05, 0.2, 0.2, 0.5]  # Trọng số cho các vùng
        distances_hsv = [self.chi2_distance(histA, histB) for histA, histB in zip(featuresA[:5], featuresB[:5])]
        distances_hog = [self.chi2_distance(histA, histB) for histA, histB in zip(featuresA[5:], featuresB[5:])]

        # distances_hsv = [self.distance_euclidean(histA, histB) for histA, histB in zip(featuresA[:5], featuresB[:5])]
        # distances_hog = [self.distance_euclidean(histA, histB) for histA, histB in zip(featuresA[5:], featuresB[5:])]
        # print(distances_hog)
        # print(distances_hsv)
        
        
        weighted_distances_hsv = [w * d for w, d in zip(weights, distances_hsv)]
        weighted_distances_hog = [w * d for w, d in zip(weights, distances_hog)]

        # # fA=[]
        # # fB=[]
        # # featuresA=list(featuresA)
        # # featuresB=list(featuresB)
        # # # print(featuresA[5])
        # # # print(featuresB[5])
        # # for i in range(5):
        # #     fA.extend(featuresA[i])
        # #     fB.extend(featuresB[i])
        # # print(len(fA), len(fB))
        # # print(len(list(featuresA[5])), len(list(featuresB[5])))
       
        # # distances_hsv = self.distance_euclidean(fA, fB)
        # # distances_hog = self.distance_euclidean(list(featuresA[5]), list(featuresB[5]))
        # # # return sum(weighted_distances_hsv) + sum(weighted_distances_hog)
        # # print(distances_hsv, distances_hog)
        # # print()
        # # return distances_hog + distances_hsv
        # lA = []
        # for item in featuresA:
        #     lA.append(list(item))

        # # print(len(lA))
        # featuresA=lA

        # fA=[]
        # fB=[]
        # featuresA=list(featuresA)
        # featuresB=list(featuresB)
        # # print(featuresA[5])
        # # print(featuresB[5])
        # for i in range(5):
        #     fA.extend(lA[i])
        #     fB.extend(featuresB[i])
        # print(len(lA))
        # print()
        # print(len(fA), len(fB))
        # print(len(list(featuresA[5])), len(list(featuresB[5])))
       
        # distances_hsv = self.distance_euclidean(fA, fB)
        # distances_hog = self.distance_euclidean(list(featuresA[5]), list(featuresB[5]))
        # # return sum(weighted_distances_hsv) + sum(weighted_distances_hog)
        # print(distances_hsv, distances_hog)
        # print()
        # return distances_hog + distances_hsv

       

        # # print(len(lB))
        # lA = []
        # for item in featuresA:
        #     lA.append(list(item))

        # print(len(lA))
        # print(len(lA))
        return sum(distances_hsv) + sum(distances_hog)

# python search.py --index index.csv --query queries/5.jpg --result-path dataset