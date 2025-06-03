from ultralytics import RTDETR

class Detector_RT(object):
    def __init__(self, weights):
        self.model = RTDETR(weights)

    def train(self, data, epochs=10, batch=16):
        print(data, epochs, batch)
        self.model.train(data=data, epochs=epochs, batch=batch)

    def detect(self, image, conf = 0.5, iou=0.9):
        return self.model(image, conf=conf, iou=iou, verbose=False)
    
