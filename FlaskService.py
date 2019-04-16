import json
import base64
import cv2
import os
import numpy as np
from Interface import meterReader
from flask import Flask, request
from locator import *
from configuration import *

app = Flask(__name__)


def getMeterNum(imageID):
    """get meter num in an image"""
    num = 0
    rootdir = templatePath
    list = os.listdir(rootdir)
    for i in range(0, len(list)):
        path = os.path.join(rootdir, list[i])
        prefix, suffix = list[i].split(".")
        prefixImageID = prefix.split("_")[0]
        if os.path.isfile(path) and suffix == "jpg" and prefixImageID == imageID:
            num += 1

    return num


def getMeterIDs(imageID):
    """get id of meters in an image"""
    meterIDs = []
    templateDir = templatePath
    list = os.listdir(templateDir)
    for i in range(0, len(list)):
        path = os.path.join(templateDir, list[i])
        prefix, suffix = list[i].split(".")
        prefixImageID = prefix.split("_")[0]
        if os.path.isfile(path) and suffix == "jpg" and prefixImageID == imageID:
            meterIDs.append(prefix)

    return meterIDs


@app.route('/', methods=['POST'])
def meterReaderAPI():
    try:
        data = request.get_data().decode("utf-8")
        data = json.loads(data)
        imageID = data["imageID"]
        path = data["path"]
        # print(imageID)
        meterIDs = getMeterIDs(imageID)

        # imageByte = data["image"].encode("ascii")
        # imageByte = base64.b64decode(imageByte)
        # imageArray = np.asarray(bytearray(imageByte), dtype="uint8")
        # image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)


        # recognitionData = None
        #
        # if path[-4:] != ".jpg":
        #     recognitionData = cv2.VideoCapture(path)
        # else:
        recognitionData = cv2.imread(path)
        # print(path, np.shape(recognitionData))
    except:
        return json.dumps({"error": "json format error!"})
    else:
        result = meterReader(recognitionData, meterIDs)
        sendData = json.dumps(result).encode("utf-8")
        return sendData


@app.route('/store', methods=['POST'])
def storeAPI():
    try:
        data = request.get_data().decode("utf-8")
        data = json.loads(data)

        imageID = data["imageID"]
        meterNum = getMeterNum(imageID)
        meterID = imageID + "_" + str(meterNum + 1)
        imageByte = data["template"].encode("ascii")
        imageByte = base64.b64decode(imageByte)
        imageArray = np.asarray(bytearray(imageByte), dtype="uint8")
        image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
        cv2.imwrite(templatePath + "/" + meterID + ".jpg", image)

        config = data["config"]
        # print(config)
    except:
        return json.dumps({"error": "json format error!"})
    else:
        file = open(configPath + "/" + meterID + ".json", "w")
        file.write(json.dumps(config))
        file.close()

        return "received!"


@app.route('/locate', methods=['POST'])
def locateAPI():
    try:
        data = request.get_data().decode("utf-8")
        data = json.loads(data)
        pointID = data["pointID"]
        path = data["path"]
        image = None

        if path[-4:] == ".jpg":
            image = cv2.imread(path)
    except:
        return json.dumps({"error":"json format error!"})
    else:
        result = locator(image, pointID)
        location = json.dumps(result).encode("utf-8")

        return location


if __name__ == '__main__':
    app.run(port=5000, debug=True)
