from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath(os.path.join(execution_path,"yolo.h5"))
detector.loadModel(detection_speed="fast")


detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path,"image.jpg"), output_image_path=os.path.join(execution_path , "imagenew.jpg"),minimum_percentage_probability=60)

print("The objects in the image are:")
for obj in detections:
    print(obj["name"])
