import numpy as np
import cv2
import time

image_BGR = cv2.imread('images/woman-working.png')

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
# Pay attention! 'cv2.imshow' takes images in BGR format
cv2.imshow('Original Image', image_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Original Image')
h, w = image_BGR.shape[:2]
blob = cv2.dnn.blobFromImage(image_BGR, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)

with open('yolo-coco-data/coco.names') as f:
    labels = [line.strip() for line in f]

print('List with labels names:')
print(labels)

network = cv2.dnn.readNetFromDarknet('yolo-coco-data/yolov4.cfg',
                                     'yolo-coco-data/yolov4.weights')

layers_names_all = network.getLayerNames()


print(layers_names_all)

layers_names_output = \
    [layers_names_all[i[0] - 1] for i in network.getUnconnectedOutLayers()]


print(layers_names_output)

probability_minimum = 0.5

threshold = 0.3

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

print(colours.shape)
print(colours[0])

network.setInput(blob)
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

# Showing spent time for forward pass
print('Objects Detection took {:.5f} seconds'.format(end - start))

bounding_boxes = []
confidences = []
class_numbers = []


for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        # Getting value of probability for defined class
        confidence_current = scores[class_current]

        if confidence_current > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))

            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                           probability_minimum, threshold)

counter = 1

if len(results) > 0:
    for i in results.flatten():
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

        # Incrementing counter
        counter += 1

        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[class_numbers[i]].tolist()

        cv2.rectangle(image_BGR, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        cv2.putText(image_BGR, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.7, colour_box_current, 2)

print()
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)

cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
# 'cv2.imshow' takes images in BGR format
cv2.imshow('Detections', image_BGR)
cv2.waitKey(0)
cv2.destroyWindow('Detections')
