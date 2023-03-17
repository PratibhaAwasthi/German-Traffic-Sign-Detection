#Import Relevant Libraries
import numpy as np
import cv2
import time
from tensorflow.keras.models import load_model
import sys


def main():
    HEIGHT = 32
    WIDTH = 32
    filename = sys.argv[1]
    darknet = cv2.dnn.readNet("yolov4_training_last.weights", "yolov4_training.cfg")

    classes = ["prohibitory", "danger", "mandatory", "other"]

    #get last layers names
    layer_names = darknet.getLayerNames()
    output_layers = [layer_names[i  - 1] for i in darknet.getUnconnectedOutLayers()]
    confidence_threshold = 0.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    start_time = time.time()
    frame_count = 0

    cap = cv2.VideoCapture(filename)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))  
    size = (frame_width, frame_height)
    font = cv2.FONT_HERSHEY_SIMPLEX
    classification_model = load_model('German_traffic_sign.h5') 
    classes_classification = ["Speed limit (20km/h)", "Speed limit (30km/h)", "Speed limit (50km/h)", "Speed limit (60km/h)", "Speed limit (70km/h)", "Speed limit (80km/h)", "End of speed limit (80km/h)", "Speed limit (100km/h)", "Speed limit (120km/h)", "No passing", "No passing for vehicles over 3.5 metric tons", "Right-of-way at the next intersection", "Priority road",  "Yield",  "Stop", "No vehicles","Vehicles over 3.5 metric tons prohibited","No entry","General caution", "Dangerous curve to the left","Dangerous curve to the right","Double curve","Bumpy road","Slippery road","Road narrows on the right","Road work","Traffic signals","Pedestrians","Children crossing","Bicycles crossing","Beware of ice/snow","Wild animals crossing","End of all speed and passing limits","Turn right ahead","Turn left ahead","Ahead only","Go straight or right","Go straight or left","Keep right","Keep left","Roundabout mandatory","End of no passing","End of no passing by vehicles over 3.5 metric tons"]

    video = cv2.VideoWriter(filename + '_output.avi', cv2.VideoWriter_fourcc(*'DIVX'),30, size)

    while True:
        ret, frame = cap.read()
        img = frame

        #get image shape
        frame_count +=1
        height, width, channels = img.shape

        # Detecting objects (YOLO)
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        darknet.setInput(blob)
        outs = darknet.forward(output_layers)

        # Showing informations on the screen (YOLO)
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > confidence_threshold:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                cv2.rectangle(img, (x, y), (x + w, y + h), (129,239,96), 2)
                crop_img = img[y:y+h, x:x+w]
                if len(crop_img) >0 and (img.all() != None):
                    crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                    crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                    prediction = np.argmax(classification_model.predict(crop_img))
                    label = str(classes_classification[prediction])
                    cv2.putText(img, label, (x, y), font, 0.5, (129,239,96), 2)

        elapsed_time = time.time() - start_time
        fps = frame_count/elapsed_time
        print ("fps: ", str(round(fps, 2)))
        cv2.imshow("Image", img)
        video.write(img)
        if cv2.waitKey(1) & 0xFF == ord ('q'):
            break
    cv2.destroyAllWindows()

main()