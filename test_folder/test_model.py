# show images inline
# %matplotlib inline

# automatically reload modules when they have changed
# %load_ext autoreload
# %autoreload 2

# import keras
import keras

# import keras_retinanet
from keras_retinanet.models.resnet import custom_objects
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)

# use this environment flag to change which GPU to use
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set the modified tf session as backend in keras
keras.backend.tensorflow_backend.set_session(get_session())

# adjust this to point to your downloaded/trained model
# model_path = os.path.join('..','keras-retinanet-master', 'snapshots', 'resnet50_pascal_02.h5')
model_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/keras-retinanet-master/snapshots/resnet50_pascal_01.h5"

# load retinanet model
model = keras.models.load_model(model_path, custom_objects=custom_objects)
# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {
    0 :     'aeroplane'   ,
    1 :     'bicycle'     ,
    2 :     'bird'        ,
    3 :     'boat'        ,
    4 :     'bottle'      ,
    5 :     'bus'         ,
    6 :     'car'         ,
    7 :     'cat'         ,
    8 :     'chair'       ,
    9 :     'cow'         ,
    10 :    'diningtable' ,
    11 :    'dog'         ,
    12 :    'horse'       ,
    13 :    'motorbike'   ,
    14 :    'person'      ,
    15 :    'pottedplant' ,
    16 :    'sheep'       ,
    17 :    'sofa'        ,
    18 :    'train'       ,
    19 :    'tvmonitor'
}

# load image
image = read_image_bgr(os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + '/test_folder/4_ppl.jpg')

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()

# _, _, boxes, nms_classification = model.predict_on_batch(np.expand_dims(image, axis=0))
_, _, detections = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)

# compute predicted labels and scores
predicted_labels = np.argmax(detections[0, :, 4:], axis=1)
scores = detections[0, np.arange(detections.shape[1]), 4 + predicted_labels]



# correct for image scale
detections[0, :, :4] /= scale

# visualize detections
for idx, (label, score) in enumerate(zip(predicted_labels, scores)):
    if score < 0.3 :
        continue

    b = detections[0, idx, :4].astype(int)
    cv2.rectangle(draw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 3)
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 0), 3)
    cv2.putText(draw, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)

plt.figure(figsize=(15, 15))
plt.axis('off')
plt.imshow(draw)
plt.show()
