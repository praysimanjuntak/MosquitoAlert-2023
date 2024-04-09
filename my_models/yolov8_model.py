import os
from ultralytics import YOLO

class YOLOModel:
    def __init__(self):
        self.model = YOLO("my_models/yolov8n-152e.onnx")
        self.classes = ['aegypti', 'albopictus', 'anopheles', 'culex', 'culiseta', 'japonicus/koreicus']

    def classify_image(self, image):
        image_information = {}
        result = self.model.predict(image, imgsz=1280, iou=0.75,  max_det=1, device="cpu")
        boxes = result[0].boxes

        if len(boxes) == 0:  # Check if no detections are made for the image
            print('No results from yolov8 model!')
        else:
            box = boxes[0]
            class_index = int(box.cpu().cls[0].item())
            box_values = box.cpu().xywhn.tolist()[0]
            image_information = {
                "class": class_index,
                "name": self.classes[class_index],
                "confidence": box.cpu().conf[0].item(),
                "xmin": box_values[0],
                "ymin": box_values[1],
                "xmax": box_values[2],
                "ymax": box_values[3]
            }

        return image_information

    def predict(self, image):
        predictedInformation = self.classify_image(image)
        mosquito_class_name_predicted = ""
        mosquito_class_bbox = [0, 0, 0, 0]

        if predictedInformation:
            mosquito_class_name_predicted = self.extract_predicted_mosquito_class_name(predictedInformation)
            mosquito_class_bbox = self.extract_predicted_mosquito_bbox(predictedInformation)

        bbox = [int(float(mcb)) for mcb in mosquito_class_bbox]

        return mosquito_class_name_predicted, bbox

    def extract_predicted_mosquito_class_name(self, extractedInformation):
        mosquito_class = ""
        if extractedInformation:
            mosquito_class = extractedInformation.get("name", "")
        return mosquito_class

    def extract_predicted_mosquito_bbox(self, extractedInformation):
        bbox = []
        if extractedInformation:
            xmin = extractedInformation.get("xmin", 0)
            ymin = extractedInformation.get("ymin", 0)
            xmax = extractedInformation.get("xmax", 0)
            ymax = extractedInformation.get("ymax", 0)
            bbox = [xmin, ymin, xmax, ymax]
        return bbox
