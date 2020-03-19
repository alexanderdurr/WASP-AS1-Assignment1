
"""
MODULE THAT DISPLAY'S PNG IMAGE AND BBOX (MANUAL INPUT FOR NAME)
"""
import matplotlib.pyplot as plt
import matplotlib.patches as patches

DIR = './yellow_box_bboxed_test/'
NAME = 'frame0000_0000'  # type name without file name ending.

img = plt.imread(DIR + NAME + ".png")
with open(DIR + NAME + ".txt", "r") as file:
	raw_string = file.readlines()[0]
	bbox_info = raw_string.split()

# RECREATE BBOX PIXEL DIMENSIONS FROM RELATIVE FLOATS ==================
x_center_p = int(float(bbox_info[1]) * img.shape[1])
y_center_p = int(float(bbox_info[2]) * img.shape[0])
width = int(float(bbox_info[3]) * img.shape[1])
height = int(float(bbox_info[4]) * img.shape[0])
top_left_xy = (x_center_p - width // 2, y_center_p - height // 2)

# Create a Rectangle patch =====================
rect = patches.Rectangle(top_left_xy, width, height, linewidth=2, edgecolor='r', facecolor='none')

# Add the patch to the Axes ======================
fig, ax = plt.subplots(1)
ax.add_patch(rect)

plt.imshow(img)
plt.show()
