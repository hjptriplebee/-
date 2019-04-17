from Algorithm.utils.Finder import meterFinderBySIFT
from Algorithm.utils.ScanPointer import scanPointer


def normalPressure(image, info):
    """
    :param image: ROI image
    :param info: information for this meter
    :return: value
    """
    meter = meterFinderBySIFT(image, info)
    result = scanPointer(meter, info)
    result = int(result * 1000) / 1000
    return [result]

