# import pytesseract
# from pytesseract import Output
# import cv2

# import Model

# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# # img = cv2.imread('12.png')
# img = cv2.imread('ac.jpeg')

# frame_extractor_obj = frameExtractor(img)
# # print(frame_extractor_obj)

# cut_digits = digits_cut.cutDigits(frame_extractor_obj)

# # print(cut_digits)
# detect = Model.Model(cut_digits)
# print(detect)
from keras.backend import set_session
from Model import Model_Multi, Model_Single, Model
import tensorflow as tf
import cv2
from frame_extractor import *
import digits_cut

session_config = tf.compat.v1.ConfigProto()
session_config.gpu_options.visible_device_list = "0"
session_config.gpu_options.allow_growth = True
set_session(tf.compat.v1.Session(config=session_config))


img = cv2.imread('ac.jpeg')

frame_extractor_obj = frameExtractor(img)
# print(frame_extractor_obj)

cut_digits = digits_cut.cutDigits(frame_extractor_obj)

model_1 = Model_Single()
model_1.train_predict()
