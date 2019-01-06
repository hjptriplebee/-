import cv2
from Common import *
import os
import json


images = os.listdir("image")
config = os.listdir("config")
for im in images:
    image = cv2.imread("image/"+im)
    print(im)
    for i in range(1, 6):
        cfg = im.split(".jpg")[0]+"_"+str(i)
        if cfg+".json" in config:
            print(cfg)
            info = json.load(open("config/"+cfg+".json"))
            x = info["ROI"]["x"]
            y = info["ROI"]["y"]
            w = info["ROI"]["w"]
            h = info["ROI"]["h"]
            cv2.rectangle(image, (x, y),
                          (x + w, y + h), (0, 0, 255), 5)

            template = cv2.imread("template/"+cfg+".jpg")
            start = np.array([info["startPoint"]["x"], info["startPoint"]["y"]])
            end = np.array([info["endPoint"]["x"], info["endPoint"]["y"]])
            center = np.array([info["centerPoint"]["x"], info["centerPoint"]["y"]])
            cv2.circle(template, (start[0], start[1]), 20, (0, 0, 255), -1)
            cv2.circle(template, (end[0], end[1]), 20, (0, 255, 0), -1)
            cv2.circle(template, (center[0], center[1]), 20, (255, 0, 0), -1)
            template = cv2.resize(template, (300, 300))
            cv2.imshow(cfg, template)
    image = cv2.resize(image, (300, 300))
    cv2.imshow(im, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

