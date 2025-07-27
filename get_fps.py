import cv2
from sys import argv

video = cv2.VideoCapture(argv[1])
print(video.get(cv2.CAP_PROP_FPS))