"""
NOTE CV DEFAULT IS BRG
"""

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from imgaug.augmenters import Sequential,SomeOf,OneOf,Sometimes,WithColorspace,WithChannels, \
            Noop,Lambda,AssertLambda,AssertShape,Scale,CropAndPad, \
            Pad,Crop,Fliplr,Flipud,Superpixels,ChangeColorspace, PerspectiveTransform, \
            Grayscale,GaussianBlur,AverageBlur,MedianBlur,Convolve, \
            Sharpen,Emboss,EdgeDetect,DirectedEdgeDetect,Add,AddElementwise, \
            AdditiveGaussianNoise,Multiply,MultiplyElementwise,Dropout, \
            CoarseDropout,Invert,ContrastNormalization,Affine,PiecewiseAffine, \
            ElasticTransformation, ChangeColorTemperature
import imageio
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import imgaug as ia
import glob
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches

NUM_IMAGES = 198
DIR_input = "./yellow_box_jpg/"
DIR_output = "./yellow_box_bboxed/"
OBJECT_CLASS = "1"  # 0: red button 1: yellow box
NUM_AUGMENTATIONS = 5
BACKGROUND_PATH = "./background/VOCdevkit/VOC2012/JPEGImages/"

# augmentation settings
seq = iaa.Sequential([
    #Sometimes(0.5, PerspectiveTransform(0.05)),
    #Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
    #Sometimes(0.5, Affine(scale=(1.0, 1.2))),
    #Sometimes(0.5, Affine(rotate=(-180, 180))),
    #Sometimes(0.5, CoarseDropout( p=0.1, size_percent=0.02) ), # Put this one in a separate layer to apply after background
    Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
    Sometimes(0.5, Add((-60, 60), per_channel=0.3)),
	#Sometimes(0.5, ChangeColorTemperature((1100, 10000), from_colorspace='RGB')),
    #Sometimes(0.3, Invert(0.2, per_channel=True)),
    Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
    Sometimes(0.5, Multiply((0.6, 1.4))),
    Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))
    ], random_order=False)

affine = iaa.Affine(
        scale=(0.5, 1.5),
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-180, 180),
        #shear=(-8, 8)
    )

occlude = iaa.Sequential([
    #Sometimes(0.5, PerspectiveTransform(0.05)),
    #Sometimes(0.5, CropAndPad(percent=(-0.05, 0.1))),
    #Sometimes(0.5, Affine(scale=(1.0, 1.2))),
    #Sometimes(0.5, Affine(rotate=(-180, 180))),
    Sometimes(0.5, CoarseDropout( p=0.1, size_percent=0.02) ),
    #Sometimes(0.5, GaussianBlur(1.2*np.random.rand())),
    Sometimes(0.5, Add((-100, 100), per_channel=0.3)),
    #Sometimes(0.3, Invert(0.2, per_channel=True)),
    #Sometimes(0.5, Multiply((0.6, 1.4), per_channel=0.5)),
    #Sometimes(0.5, Multiply((0.6, 1.4))),
    Sometimes(0.5, ContrastNormalization((0.5, 2.2), per_channel=0.3))
    ], random_order=False)

def remove_pixels_outside_bbox(img, bbox_info_short):

	# RECREATE BBOX PIXEL DIMENSIONS FROM RELATIVE FLOATS ==================
	x_center_p = int(float(bbox_info_short[0]) * img.shape[1])
	y_center_p = int(float(bbox_info_short[1]) * img.shape[0])
	width = int(float(bbox_info_short[2]) * img.shape[1])
	height = int(float(bbox_info_short[3]) * img.shape[0])
	top_left_xy = (x_center_p - width // 2, y_center_p - height // 2)

	img_new = np.zeros(img.shape, dtype=np.uint8)

	for i in range(top_left_xy[1], top_left_xy[1] + height):  # rows inside bbox
		for j in range(top_left_xy[0], top_left_xy[0] + width):  # columns inside bbox
			img_new[i, j, 0] = img[i, j, 0]
			img_new[i, j, 1] = img[i, j, 1]
			img_new[i, j, 2] = img[i, j, 2]
			img_new[i, j, 3] = img[i, j, 3]

	return img_new

def generate_bbox(bbox_prev_info, thresh, img2, img_i, check_against_prev=False):
	'''
	:param bbox_prev_info:
	:param thresh:
	:param check_against_prev:
	:return:
	'''

	matches_with_previous = False

	(major, minor, _) = cv.__version__.split(".")
	if major == 3:
		_, contours, hierarchy = cv.findContours(thresh, 1, 2)
	else:
		contours, hierarchy = cv.findContours(thresh, 1, 2)

	# compute areas of objects ===============
	areas = []
	for contour in contours:
		area = cv.contourArea(contour)
		areas.append(area)

	bbox_new_info = []
	flag_found_match = False
	num_attempts = 0

	while flag_found_match is False:
		max_index = areas.index(max(areas))
		poly = cv.approxPolyDP(contours[max_index], 3, True)
		bounding_box = cv.boundingRect(poly)  # x, y width, height
		x_center_p = (bounding_box[0] + (bounding_box[0] + bounding_box[2])) / 2  # mean of left and right borders
		y_center_p = (bounding_box[1] + (bounding_box[1] + bounding_box[3])) / 2  # mean of left and right borders
		x_center = x_center_p / img2.shape[1]
		y_center = y_center_p / img2.shape[0]
		width = bounding_box[2] / img2.shape[1]
		height = bounding_box[3] / img2.shape[0]
		bbox_new_info = [x_center, y_center, width, height]  # candidate bbox
		results = [0, 0, 0, 0]  # similarity between bbox candidate and previous one

		if img_i < 1 or check_against_prev is False:  # no mathching carried out
			bbox_prev_info = bbox_new_info
			flag_found_match = True
		else:
			# go through x_center, y_center, width, height to check similarity to previous bbox
			results[0] = abs(bbox_new_info[0] - bbox_prev_info[0])
			results[1] = abs(bbox_new_info[1] - bbox_prev_info[1])
			results[2] = abs(bbox_new_info[2] - bbox_prev_info[2])
			results[3] = abs(bbox_new_info[3] - bbox_prev_info[3])
			if sum(results) < 0.5:
				bbox_prev_info = bbox_new_info
				flag_found_match = True
			else:  # no match was found
				areas.pop(max_index)
				num_attempts += 1
				if len(areas) < 2 or num_attempts > 5:
					print("Problematic index found: " + str(img_i))
					break

	if flag_found_match is True:
		matches_with_previous = True

	return matches_with_previous, bbox_prev_info, bbox_new_info

def main():
	background_filenames = glob.glob(BACKGROUND_PATH + "*")
	num_background_images = len(background_filenames)

	print(background_filenames[0])


	bbox_prev_info = []  # previous bbox
	bbox_new_info = []  # new bbox
	for img_i in range(NUM_IMAGES):  # read images based on names
		name_no_ending = "frame" + f"{img_i:04d}"  # adds zeros to begginning of string
		# name_no_ending = entry.name[:-4]

		# READ IMAGE AND SET ALL PIXELS THAT DON'T PASS THE CONDITIONS TO 0 ==============
		img = cv.imread(DIR_input + name_no_ending + ".jpg", cv.IMREAD_UNCHANGED)
		img = cv.cvtColor(img, cv.COLOR_BGR2BGRA)  # obs!!! BGR
		img2 = np.zeros(img.shape, dtype=np.uint8)
		for i in range(img.shape[0]):
			for j in range(img.shape[1]):

				# # RED BUTTON CONDITION (1. if not enough red. 2. if too intense) =========================
				# if (img[i, j, 0] > 30 and img[i, j, 1] > 25 and img[i, j, 2] < 150) or \
				# 		(img[i, j, 0] > 200 and img[i, j, 1] > 200 and img[i, j, 2] > 200):

				# YELLOW BUTTON CONDITION (just intensity) =========================
				if (img[i, j, 0] < 150 and img[i, j, 1] < 150 and img[i, j, 2] < 150):
					img2[i, j, 0] = 0  # B
					img2[i, j, 1] = 0  # G
					img2[i, j, 2] = 0  # R
					img2[i, j, 3] = 0  # Alpha
				else:
					img2[i, j, 0] = img[i, j, 0]
					img2[i, j, 1] = img[i, j, 1]
					img2[i, j, 2] = img[i, j, 2]
					img2[i, j, 3] = img[i, j, 3]

		# open =============
		kernel = np.ones((2, 2), np.uint8)
		img2 = cv.erode(img2, kernel, iterations=1)
		img2 = cv.dilate(img2, kernel, iterations=1)

		# create contours ===================
		img_gray = cv.cvtColor(img2, cv.COLOR_RGB2GRAY)
		ret, thresh = cv.threshold(img_gray, 1, 255, 0)

		matches_with_previous, bbox_prev_info, bbox_new_info = generate_bbox(bbox_prev_info, thresh, img2, img_i, check_against_prev=True)

		if matches_with_previous:  # a candidate bbox was found which matches with previous one
			img2 = remove_pixels_outside_bbox(img2, bbox_new_info)
			with open(DIR_output + name_no_ending + ".txt", "w") as file:
				file.write(OBJECT_CLASS + " " + str(bbox_new_info[0]) + " " +
						   str(bbox_new_info[1]) + " " +
						   str(bbox_new_info[2]) + " " +
						   str(bbox_new_info[3]))

			cv.imwrite(DIR_output + name_no_ending + ".png", img2)

			# AUGMENTATION HERE
			for i in range(NUM_AUGMENTATIONS):
				# rotational augmentations ===============
				img3 = np.copy(img2)
				img3 = affine(image=img3)
				img_gray = cv.cvtColor(img3, cv.COLOR_RGB2GRAY)
				ret, thresh = cv.threshold(img_gray, 1, 255, 0)

				_, _, bbox_new_info = generate_bbox([], thresh, img3, img_i, check_against_prev=False)  # bbox_prev_info won't be used

				# plot for debug
				debug = False
				if debug:
					f, axarr = plt.subplots(2,1)
					axarr[0].imshow(img3)

				# object coloration augmentations ===================
				temp = seq(image=img3)
				img3[:, :, 0:3] = temp[:, :, 0:3]

				if debug:
					axarr[1].imshow(img3)
					plt.show()

				# foreground and background prep ===================
				# loop until you find an image that is big enough?
				#while True:
				back_path = background_filenames[np.random.randint(num_background_images)]
				img_back = cv.imread(back_path, cv.IMREAD_UNCHANGED)
				img_back = cv.cvtColor(img_back, cv.COLOR_BGR2BGRA)  # obs!!! BGR

				# resize background to fit image size
				H,W = img_back.shape[:2]
				y_anchor = int(np.random.rand() * (H-img3.shape[0]))
				x_anchor = int(np.random.rand() * (W-img3.shape[1]))
				# img_back = cv2.resize(img_back, img2.shape[:2])
				#img_back = img_back[y_anchor:y_anchor+img3.shape[0],x_anchor:x_anchor+img3.shape[1],:]
				img_back = cv2.resize(img_back, (img3.shape[1], img3.shape[0]))


				# foreground and background augmentations ===================
				# normalize alpha mask to 0-1, and use to make new image
				alpha = img3[:, :, 0:3].astype(float)
				for j in range(3):
					alpha[:, :, j] = img3[:, :, 3].astype(float)/255
				img4 = img3[:, :, 0:3] * alpha + img_back[:, :, 0:3] * (1 - alpha)
				img4 = img4.astype(np.uint8)

				img4 = occlude(image=img4)
				temp = seq(image=img3)

				# save augmented with extra index =======
				with open(DIR_output + name_no_ending +"_" + f"{i:04d}" + ".txt", "w") as file:
					file.write(OBJECT_CLASS + " " + str(bbox_new_info[0]) + " " +
							   str(bbox_new_info[1]) + " " +
							   str(bbox_new_info[2]) + " " +
							   str(bbox_new_info[3]))

				cv.imwrite(DIR_output + name_no_ending +"_" + f"{i:04d}" + ".png", img4)

				# make empty version as well ============
				img_back2 = occlude(image=img_back)
				img_back2 = seq(image=img_back2)
				img_back2[:, :, 3] = img_back[:, :, 3]

				with open(DIR_output + name_no_ending +"_" + f"{i:04d}" + "_empty.txt", "w") as file:
					file.write("")

				cv.imwrite(DIR_output + name_no_ending +"_" + f"{i:04d}" + "_empty.png", img_back2)

		if img_i % 5 == 0:
			print(img_i)

		# if img_i > 3:
		# 	break

if __name__ == '__main__':
    main()
