import os

# os.system("cd test_folder & python test.py")
snapshot_path = "C:/Users/JeallyBeans/Dropbox/3aar_elektro_bachelor/Bach-prosjekt/Retinanet/keras-retinanet-master/snapshots/resnet50_pascal_07.h5"

config_train = "--snapshot "+ snapshot_path + "  --freeze-backbone --epochs 13 --steps 1000 --no-evaluation "
config_debug = "--anchors --annotations "

#os.system("cd keras-retinanet-master & " +
#         "cd &"                          +
#        "keras_retinanet\\bin\\debug.py " +
#         config_debug                    +
#        "pascal C:/Users/JeallyBeans/Dropbox/3aar_elektro_bachelor/Bach-prosjekt/Retinanet/VOC_2007_2012/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007")


os.system("cd keras-retinanet-master & " +
         "cd &"                           +
        "keras_retinanet\\bin\\train.py " +
        config_train                      +
        "pascal C:/Users/JeallyBeans/Dropbox/3aar_elektro_bachelor/Bach-prosjekt/Retinanet/VOC_2007_2012/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007")
