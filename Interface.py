import json
import os
import cv2
from Algorithm.absorb import absorb
from Algorithm.Blenometer import checkBleno
from Algorithm.SF6 import SF6Reader
from Algorithm.oilTempreture import oilTempreture
from Algorithm.videoDigit import videoDigit

from Algorithm.arrest.countArrester import countArrester
from Algorithm.arrest.doubleArrester import doubleArrester

from Algorithm.pressure.digitPressure import digitPressure
from Algorithm.pressure.normalPressure import normalPressure
from Algorithm.pressure.colorPressure import colorPressure


from Algorithm.onoff.contactStatus import contactStatus
from Algorithm.onoff.onoffIndoor import onoffIndoor
from Algorithm.onoff.onoffOutdoor import onoffOutdoor
from Algorithm.onoff.onoffBatteryScreen import onoffBattery
from Algorithm.onoff.readyStatus import readyStatus
from Algorithm.onoff.springStatus import springStatus

from Algorithm.others.colorIndicator import colorIndicator

from configuration import *


def meterReaderCallBack(image, info):
    """call back function"""
    if info["type"] == None:
        return "meter type not support!"
    else:
        return info["type"](image, info)


def getInfo(ID):
    """
    get info from file
    :param ID: meter ID
    :return: info = {
            "distance": 10,
            "horizontal": 10,
            "vertical": 20,
            "name": "1_1",
            "type": SF6,
            "template": "template.jpg",
            "ROI": {
                "x": 200,
                "y": 200,
                "w": 1520,
                "h": 680
            },
            "startPoint": {
                "x": -1,
                "y": -1
            },
            "endPoint": {
                "x": -1,
                "y": -1
            },
            "centerPoint": {
                "x": -1,
                "y": -1
            },
            "startValue": 0,
            "totalValue": 2
        }
    """
    file = open(configPath + "/" + ID + ".json")
    info = json.load(file)
    # string to pointer
    if info["type"] == "absorb":
        info["type"] = absorb
    elif info["type"] == "digitPressure":
        info["type"] = digitPressure
    elif info["type"] == "normalPressure":
        info["type"] = normalPressure
    elif info["type"] == "contactStatus":
        info["type"] = contactStatus
    elif info["type"] == "colorPressure":
        info["type"] = colorPressure
    elif info["type"] == "SF6":
        info["type"] = SF6Reader
    elif info["type"] == "countArrester":
        info["type"] = countArrester
    elif info["type"] == "doubleArrester":
        info["type"] = doubleArrester
    elif info["type"] == "oilTempreture":
        info["type"] = oilTempreture
    elif info["type"] == "blenometer":
        info["type"] = checkBleno
    elif info["type"] == "onoffIndoor":
        info["type"] = onoffIndoor
    elif info["type"] == "onoffOutdoor":
        info["type"] = onoffOutdoor
    elif info["type"] == "onoffBattery":
        info["type"] = onoffBattery
    elif info["type"] == "videoDigit":
        info["type"] = videoDigit
    elif info["type"] == "readyStatus":
        info["type"] = readyStatus
    elif info["type"] == "springStatus":
        info["type"] = springStatus
    elif info["type"] == "colorIndicator":
        info["type"] = colorIndicator
    else:
        info["type"] = None
    info["template"] = cv2.imread(templatePath + "/" + ID + ".jpg")
    if info["digitType"] != "False":
        info.update(json.load(open(os.path.join("ocr_config", info["digitType"] + ".json"))))
    return info


def meterReader(recognitionData, meterIDs):
    """
    global interface
    :param recognitionData: image or video
    :param meterIDs: list of meter ID
    :return:
    """
    # results = {}
    results = []
    for i, ID in enumerate(meterIDs):
        # get info from file
        info = getInfo(ID)
        if info["digitType"] == "VIDEO":
            results[ID] = meterReaderCallBack(recognitionData, info)
        else:
            # ROI extract
            x = info["ROI"]["x"]
            y = info["ROI"]["y"]
            w = info["ROI"]["w"]
            h = info["ROI"]["h"]
            # call back
            # cv2.rectangle(recognitionData, (x, y), (x+w, y + h), (255, 0, 0), 3)
            # cv2.imshow("testInput", recognitionData)
            # cv2.waitKey(0)
            if x != 0 or y != 0 or w != 0 or h != 0:
                ROI = recognitionData[y:y + h, x:x + w]
                results.append(meterReaderCallBack(ROI, info))
            else:
                results.append(meterReaderCallBack(recognitionData, info))
    return results