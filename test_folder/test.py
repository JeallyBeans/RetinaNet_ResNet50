import os

#cur_path = os.path.dirname(__file__)
#print(cur_path)
#fpath = os.path.join(direct, "test2.txt")

#file = open(, "r")
parentDir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
model_path = os.path.abspath(os.path.join(os.getcwd(), os.pardir)) + "/keras-retinanet-master/snapshots/"
absDir = os.pardir
print("--------- model_path---------- \n " + model_path)
print("--------- absDir---------- \n " + absDir)


#os.system("cd keras-retinanet-master & dir")

# print (file.read())
