# MITRA-Multi-Terrain-Agricultural-Bot
MITRA is an autonomous robot capable of navigation using computer vision, localization, 3D mapping, and Sensor Data Management to navigate around the field. It is capable to identify weeds and deployed with a mechanism to remove them. Also, it aims to help farmers in the irrigation of crops and guard the crop against pests.

## METHODOLOGY
Agricultural robots for weed picking are designed with several components. Image acquisition and processing involve capturing and filtering the image to get a clear image of the crop for accurate prediction. Image segmentation extracts the region of interest using Convolutional Neural Networks. Image classification uses feature values and a CNN algorithm to classify the image as a weed or crop. Our solution involves building an autonomous rover capable of autonomous navigation in the field and self-identifying the weeds among the crops and removing them with the help of a robotic arm.
<img src="IP/weed_detect.jpg">
## WORKING
The mechanical design includes a lightweight frame with high load capacity, honeycomb wheels for rough terrain, and an end-effector for harvesting with three parts: upper and middle plates for cutting and punching holes in the weed and a lower part with a cutting device. The manipulator contains four-bar parallel links to maintain the gripper's position and height for navigation through the environment.
