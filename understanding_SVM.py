#This script has been solely created under dataturks. Copyrights are reserved

#EXAMPLE USAGE
#python3 keras_json_parser.py --json_file "flower.json" --dataset_path "Dataset5/" --train_percentage 80 --validation_percentage 20


import json 
import glob
import urllib.request
import argparse
import random
import os



def downloader(image_url , i):
    file_name = str(i)
    full_file_name = str(file_name) + '.jpg'
    urllib.request.urlretrieve(image_url,full_file_name)


if __name__ == "__main__":
	a = argparse.ArgumentParser()
	a.add_argument("--json_file", help="path to json")
	a.add_argument("--dataset_path", help="path to the dataset")
	a.add_argument("--train_percentage", help="percentage of data for training")
	a.add_argument("--validation_percentage", help="percentage of data for validation")

	args = a.parse_args()

	if args.json_file is None and args.dataset_path is None and args.train_percentage is None and args.validation_percentage is None:
	  a.print_help()
	  sys.exit(1)


	with open(args.json_file) as file1:
		lis = []	
		for i in file1:
			lis.append(json.loads(i))

	train = float(int(args.train_percentage) / 100)
	test = float(int(args.validation_percentage)  / 100)

	folder_names = []
	label_to_urls = {}

	for i in lis:
		if i['annotation']['labels'][0] not in folder_names:
			folder_names.append(i['annotation']['labels'][0])
			label_to_urls[i['annotation']['labels'][0]] = [i['content']] 	
		else:
			label_to_urls[i['annotation']['labels'][0]].append(i['content'])

	print(label_to_urls.keys())

	os.mkdir(args.dataset_path)
	os.chdir(args.dataset_path)
	os.mkdir("train")
	os.mkdir("validation")


	os.chdir("train/")
	for i in label_to_urls.keys():
		os.mkdir(str(i))
		os.chdir(str(i))	
		k = 0	
		for j in label_to_urls[i][:round(train*(len(label_to_urls[i])))]:
			#print(label_to_urls[i][:round(0.8*(len(label_to_urls[i])))])		
			downloader(j , str(i)+str(k))
			k+=1
		os.chdir("../")
	print(os.getcwd())
	os.chdir("../validation/")
	for i in label_to_urls.keys():
		os.mkdir(str(i))
		os.chdir(str(i))	
		k = 0	
		for j in label_to_urls[i][round(0.8*(len(label_to_urls[i]))):]:
			downloader(j , str(i)+str(k))
			k+=1
		os.chdir("../")




def fd_hu_moments(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature
def fd_haralick(image):    # convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the haralick texture feature vector
    haralick = mahotas.features.haralick(gray).mean(axis=0)
    return haralick
 
def fd_histogram(image, mask=None):
    # convert the image to HSV color-space
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # compute the color histogram
    hist  = cv2.calcHist([image], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    # normalize the histogram
    cv2.normalize(hist, hist)
    hist.flatten()


global_feature = np.hstack([fd_histogram(image), fd_haralick(image), fd_hu_moments(image)])
scaler = MinMaxScaler(feature_range=(0, 1))
#Normalize The feature vectors...
rescaled_features = scaler.fit_transform(global_features)


from sklearn.svm import SVC
clf = models.append(('SVM', SVC(random_state=9)))
prediction= clf.fit(global_feature.reshape(1,-1))[0]