import os as os

Debug = False

# Directory where the desired snapshot is stored.
# os.path.abspath... Creates the path from the parent folder to where this file is stored.
snapshot_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/keras-retinanet-master/snapshots/resnet50_pascal_07.h5"

# Directory to where the data set is stored. Functions the same as snapshot_path over---^
data_path     =  os.path.abspath(os.path.join(os.getcwd(), os.pardir)) +  "/VOC_2007_2012/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007")

# Arguments to pass along when running the train.py file from keras-retinanet-master repository.
# More arguments can be found in the train.py file.
config_train = "--snapshot " + snapshot_path + "  --freeze-backbone --epochs 13 --steps 1000 --no-evaluation "

# Arguments to pass along the debug.py file. Has the same parameter as the above Str.
config_debug = "--anchors --annotations "

if Debug = True:
    os.system("cd keras-retinanet-master & " +
        "cd &"                               +
       "keras_retinanet\\bin\\debug.py "     +
        config_debug                         +
       "pascal " + data_path)

else:
    os.system("cd keras-retinanet-master & " +
         "cd &"                              +
        "keras_retinanet\\bin\\train.py "    +
        config_train                         +
        "pascal " + data_path)
