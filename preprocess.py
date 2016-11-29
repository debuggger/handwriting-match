from PIL import Image
import os
import numpy as np
import itertools
import random
import pickle

class Preprocess:
    def __init__(self, dir, newsize):
        self.dir = dir
	self.image1DArray = []
	self.image2DArray = []
        self.labels = []
        self.newsize = newsize

    def preprocess(self):
        np.set_printoptions(threshold=np.inf)
        print self.newsize
        for file in os.listdir(self.dir):
            if file.endswith('.png'):
                try:
                    img = Image.open(self.dir+'/'+file)
                    img2D = np.asarray(img.resize(self.newsize).getdata(), dtype=np.float16).reshape(self.newsize[::-1])/255.0
                    img1D = np.asarray(img.resize(self.newsize).getdata())/255.0
        
                    self.image1DArray.append(img1D)
                    self.image2DArray.append(img2D)
                    self.labels.append(file[:4])
                except:
                    print("Exception:"+file)

    def preprocessTest(self, img1, img2):
	out = []
        image1 = Image.open(img1)
        image2 = Image.open(img2)
        img1_2D = np.asarray(image1.resize(self.newsize).getdata(), dtype=np.float16).reshape(self.newsize[::-1])/255.0
        img1_1D = np.asarray(image1.resize(self.newsize).getdata())/255.0
        img2_2D = np.asarray(image2.resize(self.newsize).getdata(), dtype=np.float16).reshape(self.newsize[::-1])/255.0
        img2_1D = np.asarray(image2.resize(self.newsize).getdata())/255.0
	out.append(np.hstack((img1_1D, img2_1D)))

	return out

    def generateTrainingSamples(self, data, label):
        train_x = []
        train_y = []
        count = 0
	dataLabelMap = {}

	for i in range(len(label)):
		if not dataLabelMap.has_key(label[i]):
			dataLabelMap[label[i]] = []
		dataLabelMap[label[i]].append(data[i])

        for i in dataLabelMap:
            n = len(dataLabelMap[i])
            matchCombinations = itertools.combinations(range(n), 2)
            
            for match in matchCombinations:
                train_x.append(np.hstack((dataLabelMap[i][match[0]], dataLabelMap[i][match[1]])))
                train_y.append(np.array([0, 1]))
                train_x.append(np.hstack((dataLabelMap[i][match[1]], dataLabelMap[i][match[0]])))
                train_y.append(np.array([0, 1]))

            mismatches = dataLabelMap.keys()[:]
            mismatches.remove(i)

            for j in range(len(dataLabelMap[i])):
                for mismatch in np.random.choice(mismatches, int((n-1)/2), False):
                    randomIndex = random.randint(0, len(dataLabelMap[mismatch])-1)
                    train_x.append(np.hstack((dataLabelMap[i][j], dataLabelMap[mismatch][randomIndex])))
                    train_y.append(np.array([1, 0]))
                for mismatch in np.random.choice(mismatches, int((n-1)/2), False):
                    randomIndex = random.randint(0, len(dataLabelMap[mismatch])-1)
                    train_x.append(np.hstack((dataLabelMap[mismatch][randomIndex], dataLabelMap[i][j])))
                    train_y.append(np.array([1, 0]))
            count += 1
        return train_x, train_y
                

