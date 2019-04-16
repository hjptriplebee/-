import requests
import base64
import json
import os
import time
import cv2
import multiprocessing
from configuration import *

from Interface import meterReader


def startServer():
    os.system("python FlaskService.py")


def startClient(results):
    images = os.listdir("info/20190410/IMAGES/Pic_2")
    for im in images:
        path = "info/20190410/IMAGES/Pic_2/" + im
        data = json.dumps({
            "path": path,
            "imageID": im.split('.')[0] + "_1"
        })
        print(path, im)
        print(data)
        r = requests.post("http://127.0.0.1:5000/", data=data.encode("utf-8"))
        print(r.text)
        receive = json.loads(r.text)
        print(im, receive)

        results.append(True)


def testReadyStatus():
    imgPath = "image"
    configPath = "info/20190410/config"
    images = os.listdir(imgPath)
    config = os.listdir(configPath)
    for im in images:
        filename, extention = os.path.splitext(im.lower())
        if extention == '.jpg' or extention == '.png':
            image = cv2.imread(imgPath + "/" + im)
            for i in range(1, 6):
                cfg = filename + "_" + str(i)
                if cfg + ".json" in config:
                    receive2 = meterReader(image, [cfg])
                    print(cfg, receive2)


def codecov():
    images = os.listdir("image")
    config = os.listdir("config")

    for im in images:
        image = cv2.imread(imgPath + "/" + im)
        print(im)
        pos = im.split(".")[0].split("-")

        for i in range(1, 6):
            cfg = pos[0] + "-" + pos[1] + "_" + str(i)
            if cfg + ".json" in config:
                receive2 = meterReader(image, [cfg])
                print(cfg, receive2)

    print("codecov done")


def testVideo():
    video_path = ("video_")
    config = os.listdir("config")

    for file in os.listdir(video_path):
        if file.startswith(".DS"):
            continue
        video = cv2.VideoCapture(os.path.join(video_path, file))
        result = meterReader(video, [file[:-4] + "_1"])
        print(file, result)
    print("codecov done")


if __name__ == "__main__":
    # serverProcess = multiprocessing.Process(target=startServer)
    # results = multiprocessing.Manager().list()
    # clientProcess = multiprocessing.Process(target=startClient, args=(results,))
    # serverProcess.start()
    # time.sleep(30)
    # clientProcess.start()
    # clientProcess.join()
    # serverProcess.terminate()
    # testReadyStatus()
    codecov()
    # testVideo()
    #
    # for i in range(20):
    #     serverProcess = multiprocessing.Process(target=startServer)
    #     results = multiprocessing.Manager().list()
    #     clientProcess = multiprocessing.Process(target=startClient, args=(results,))
    #     serverProcess.start()
    #     time.sleep(30)
    #     clientProcess.start()
    #     clientProcess.join()
    #     serverProcess.terminate()
    #
    #     codecov()

    # for result in results:
    #     print(result)
    #     if not result:
    #         exit(100)
