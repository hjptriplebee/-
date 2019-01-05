import socket
import threading
import json
import base64
import time
import multiprocessing
import cv2
import numpy as np
from Interface import meterReader

class meterReaderService:
    """TCP service"""
    def __requestHandler(self, clientSocket, clientAddress):
        """
        :param clientSocket: client socket
        :param clientAddress: client address
        :return: 
        """
        print('Accept new connection from %s:%s...' % clientAddress)
        # collect buffer
        receiveData = ""
        bufferSize = 2048
        while True:
            buffer = clientSocket.recv(bufferSize)
            # when client close socket, not the end of data
            if not buffer:
                break
            receiveData += buffer.decode("utf-8")
            # when
            if len(buffer) != bufferSize and receiveData[-1] == '}':
                break

        # feed data to interface
        data = json.loads(receiveData)
        meterIDs = data["meterIDs"]
        imageByte = data["image"].encode("ascii")
        imageByte = base64.b64decode(imageByte)
        imageArray = np.asarray(bytearray(imageByte), dtype = "uint8")
        image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)

        result = meterReader(image, meterIDs)
        sendData = json.dumps(result).encode("utf-8")
        clientSocket.send(sendData)
        print("Result is sent to client!")
        clientSocket.close()
        print('Close new connection from %s:%s...' % clientAddress)

    def startServer(self, port = 9999, maxConnection = 5):
        """
        start service
        :param port: port
        :param maxConnection: max connection at a time
        """
        self.serverSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.serverSocket.bind(("127.0.0.1", port))
        self.serverSocket.listen(maxConnection)
        print("Binding meter reader service to port %d." % port)
        print("Waiting for connection...")

        # listen
        while True:
            clientSocket, clientAddress = self.serverSocket.accept()
            # create a new thread th handle connection
            thread = threading.Thread(target = self.__requestHandler, args = (clientSocket, clientAddress))
            thread.start()

    def startClient(self, port = 9999):
        """
        test service
        :param port: port
        """
        self.clientSocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientSocket.connect(("127.0.0.1", port))

        # image to byte
        image = open("image/2018-11-20-16-22-02.jpg", "rb")
        imageByte = base64.b64encode(image.read())
        data = {
            "image": imageByte.decode("ascii"),
            "meterIDs": [
                "1_1"
            ]
        }

        sendData = json.dumps(data).encode("utf-8")
        self.clientSocket.send(sendData)
        print("data is sent to server!")
        print(self.clientSocket.recv(1024).decode("utf-8"))
        self.clientSocket.close()


if __name__ == "__main__":
    test = meterReaderService()
    # # serverProcess = multiprocessing.Process(target = test.startServer)
    # # clientProcess = multiprocessing.Process(target=test.startClient)
    # # serverProcess.start()
    # # time.sleep(3)
    # # clientProcess.start()
    image = cv2.imread("image/bileiqi1.JPG")
    print(meterReader(image, ["bileiqi1_1"]))



