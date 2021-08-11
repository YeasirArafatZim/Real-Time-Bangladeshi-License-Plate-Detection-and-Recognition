import numpy as np
import os
import tensorflow as tf
from utils import label_map_util
from utils import visualization_utils as vis_util
import natsort
import torch
import torch.nn as nn
import cv2
import glob
from datetime import datetime
from csv import writer

cap = cv2.VideoCapture('test.mp4')

# What model to download.
MODEL_NAME = 'new_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('numberPlate_training', 'labelmap.pbtxt')

NUM_CLASSES = 3


# Deep Neural Network Model for segmented image recognition
class DeepNeuralNetworkModel(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        # 1st hidden layer
        self.linear_1 = nn.Linear(input_size, 1000)
        # Non-linearity in 1st hidden layer
        self.relu_1 = nn.LeakyReLU()

        # 2nd hidden layer
        self.linear_2 = nn.Linear(1000, 500)
        # Non-linearity in 2nd hidden layer
        self.relu_2 = nn.LeakyReLU()

        # 3rd hidden layer
        self.linear_3 = nn.Linear(500, 128)
        # Non-linearity in 2nd hidden layer
        self.sigmoid_3 = nn.Sigmoid()

        self.linear_out = nn.Linear(128, num_classes)

    def forward(self, x):
        # 1st hidden layer
        out = self.linear_1(x)
        # Non-linearity in 1st hidden layer
        out = self.relu_1(out)

        # 2nd hidden layer
        out = self.linear_2(out)
        # Non-linearity in 2nd hidden layer
        out = self.relu_2(out)

        # 3rd hidden layer
        out = self.linear_3(out)
        # Non-linearity in 3rd hidden layer
        out = self.sigmoid_3(out)

        probas = self.linear_out(out)

        return probas


# Defining Neural Network Model
segmentationModel = DeepNeuralNetworkModel(input_size=3000, num_classes=13)
# Loading trained segmentationModel
segmentationModel.load_state_dict(torch.load('new_graph/templateMatching.pkl', map_location="cpu"))


def append_list_as_row(file_name, list_of_elem):
    with open(file_name, 'a', newline='') as write_obj:
        csv_writer = writer(write_obj)
        csv_writer.writerow(list_of_elem)


# Read the model from the file
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
        (im_height, im_width, 3)).astype(np.uint8)


def pre_process(image):
    # pre-processing
    global crop
    img = cv2.resize(image, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    temp_img = cv2.resize(image, dsize=(222, 118), interpolation=cv2.INTER_CUBIC)

    # Removing older preprocessed images
    files = glob.glob('temp\\preprocess\\*.png')
    for file in files:
        os.remove(file)

    # Sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    img = cv2.filter2D(img, -1, kernel)

    # Denoise
    img = cv2.fastNlMeansDenoising(img, None, 20, 15, 3)

    ad = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 55, 1)  # (55, 1)

    im = cv2.resize(ad, dsize=(222, 118), interpolation=cv2.INTER_CUBIC)
    # adding border
    row, col = im.shape
    im = cv2.rectangle(im, (0, 0), (col, row), (255, 255, 255), 6)

    # Morphological Closing
    kernel = np.ones((3, 3), np.uint8)
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

    # threshold image
    ret1, threshed_img = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

    # find contours and get the external one
    contours, heir = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # with each contour, draw boundingRect
    for c in contours:
        if 10000 < cv2.contourArea(c) < 20000:
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            # draw a green rectangle to visualize the bounding rect
            temp_img = cv2.rectangle(temp_img, (x, y), (x + w, y + h), (200, 200, 200), 20)
            crop = temp_img[y: y + h, x: x + w]
            cv2.imwrite('temp/preprocess/cropped.png', crop)
    return crop


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b: b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return cnts


def horizontal_seg(img):
    ad = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 30)

    # Denoise
    im = cv2.fastNlMeansDenoising(ad, None, 20, 15, 3)
    im = cv2.GaussianBlur(im, (5, 5), 0)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(17, 1))
    ad = cv2.erode(im, kernel)

    # threshold image
    ret, threshed_img = cv2.threshold(ad, 127, 255, cv2.THRESH_BINARY)

    # find contours and get the external one
    contours, heir = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sorting contours from top-to-bottom
    contours = sort_contours(contours, "top-to-bottom")
    count = 0
    # with each contour, draw boundingRect
    for c in contours:
        if 2000 < cv2.contourArea(c) < 5000:
            count += 1
            # get the bounding rect
            x, y, w, h = cv2.boundingRect(c)
            crop = img[y: y + h, x: x + w]
            cv2.imwrite('temp/preprocess/h_seg' + str(count) + '.png', crop)


def vertical_seg():
    # Removing older segmented images
    files = glob.glob('temp\\segmentation\\*.png')
    for file in files:
        os.remove(file)

    check = os.path.isfile("temp\\preprocess\\h_seg1.png")
    if check == True:
        # Segmentation for 1st image
        img = cv2.imread('temp\\preprocess\\h_seg1.png', 0)

        temp_img = cv2.resize(img, dsize=(180, 30), interpolation=cv2.INTER_CUBIC)

        ad = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 20)  # (55, 1)

        # Erosion
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(5, 1))
        im = cv2.erode(ad, kernel)

        # Adding border
        row, col = im.shape
        im = cv2.rectangle(im, (0, 0), (col, row), (255, 255, 255), 6)

        im = cv2.resize(im, dsize=(180, 30), interpolation=cv2.INTER_CUBIC)

        # threshold image
        ret, threshed_img = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

        # find contours and get the external one
        contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sort_contours(contours)

        # with each contour, draw boundingRect in green
        count = 0
        for c in contours:
            if 200 < cv2.contourArea(c) < 1500:
                count += 1
                # get the bounding rect
                x, y, w, h = cv2.boundingRect(c)
                seg = temp_img[y: y + h, x: x + w]
                cv2.imwrite('temp/segmentation/seg' + str(count) + '.png', seg)

    check = os.path.isfile("temp\\preprocess\\h_seg2.png")
    if check == True:
        # Segmentation for 2nd image
        img = cv2.imread('temp\\preprocess\\h_seg2.png', 0)
        temp_img = cv2.resize(img, dsize=(180, 30), interpolation=cv2.INTER_CUBIC)

        ad = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 55, 25)  # (55, 1)

        ad = cv2.resize(ad, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(1, 1))
        im = cv2.erode(ad, kernel)

        # Adding border
        row, col = im.shape
        im = cv2.rectangle(im, (0, 0), (col, row), (255, 255, 255), 6)

        im = cv2.resize(im, dsize=(180, 30), interpolation=cv2.INTER_CUBIC)

        # threshold image
        ret, threshed_img = cv2.threshold(im, 127, 255, cv2.THRESH_BINARY)

        # find contours and get the external one
        contours, hier = cv2.findContours(threshed_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sort_contours(contours)

        # with each contour, draw boundingRect in green
        for c in contours:
            if 180 < cv2.contourArea(c) < 800:
                count += 1
                # get the bounding rect
                x, y, w, h = cv2.boundingRect(c)
                seg = temp_img[y: y + h, x: x + w]
                cv2.imwrite('temp/segmentation/seg' + str(count) + '.png', seg)


def recognition():
    # Loading segmented images
    images = glob.glob('temp\\segmentation\\*.png')
    # sorting the list by their name
    images = natsort.natsorted(images)
    images = [cv2.imread(im, 0) for im in images]
    images = [torch.Tensor(cv2.resize(img, dsize=(100, 30), interpolation=cv2.INTER_CUBIC)) for img in images]

    string = ''
    for img in images:
        predictions = segmentationModel.forward(img.view(-1, 100*30))
        prediction = torch.argmax(predictions, dim=1).numpy()
        if prediction[0] == 10:
            string = string + 'Dhaka'
        elif prediction[0] == 11:
            string = string + 'Metro'
        elif prediction[0] == 12:
            string = string + 'Ga-'
        else:
            string = string + str(prediction[0])
    # Current Date and Time
    now = datetime.now()
    now = now.strftime("%d-%b-%Y %H:%M:%S")
    append_list_as_row('temp/segmentation/numberPlateList.csv', [string, now])
    return string


def process_img2string(img, t):
    # Resize the image with interpolation
    cv2.imwrite("temp/number_plate_detected" + str(t) + ".png", img)
    # gray conversion
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Region of Interest selection
    img = pre_process(img)
    # Horizontally segmentation using image processing
    horizontal_seg(img)
    # Vertical Segmentation of those image
    vertical_seg()
    # Recognition of segmented  images
    string = recognition()
    return string


te = 0
with detection_graph.as_default():
    with tf.compat.v1.Session(graph=detection_graph) as sess:
        while True:
            ret, image_np = cap.read()
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            out = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})

            rows = image_np.shape[0]
            cols = image_np.shape[1]

            # Visualization of the results of a detection.
            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np,
                np.squeeze(out[0][0]),
                np.squeeze(out[2][0]).astype(np.int32),
                np.squeeze(out[1][0]),
                category_index,
                use_normalized_coordinates=True,
                line_thickness=4)
            num_detections = int(out[3][0])

            for i in range(num_detections):
                classId = int(out[2][0][i])
                score = float(out[1][0][i])
                bbox = [float(v) for v in out[0][0][i]]

                if score > 0.9 and classId == 2:
                    # Creating a box around the detected number plate
                    x = int(bbox[1] * cols)
                    y = int(bbox[0] * rows)
                    right = int(bbox[3] * cols)
                    bottom = int(bbox[2] * rows)
                    # Extract the detected number plate
                    tmp = image_np[y: bottom, x: right]
                    te = te + 1
                    text = process_img2string(tmp, te)
                    text_height = 1
                    cv2.rectangle(image_np, (x, y), (right, bottom), (0, 255, 255), thickness=2)
                    cv2.putText(image_np, text, (x, bottom + 19),
                                cv2.FONT_HERSHEY_SIMPLEX, text_height, (0, 255, 255), 2)
                    cv2.imwrite('temp/full_image.png', image_np)

            # print(category_index[2]['name'])
            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            if cv2.waitKey(25) & 0xFF == ord('q'):
                cv2.destroyAllWindows()
                break
