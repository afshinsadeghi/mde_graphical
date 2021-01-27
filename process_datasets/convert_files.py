# in all file replace space with tab
#in dic file replace colum 0 and 1
# in test train valid file replace column number 1 and 2 (second and third column)
import numpy as np
import os


def covert_files_in_folder(path, output_path):
    print("processing :", path)
    entity2id = np.loadtxt(open(path + "entity2id.txt", "rb"), delimiter="\t",skiprows=1, dtype="str")
    #np.savetxt("entity2id-dic-old.txt",entity2id,fmt='%s')
    entity2id_ = entity2id.copy()#np.copy(entity2id)#np.zeros(shape=entity2id.shape, dtype=str)
    entity2id_[:,0]= entity2id[:,1]
    entity2id_[:,1]= entity2id[:,0]
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    np.savetxt(output_path + "entities.dict",entity2id_,fmt='%s',delimiter="\t")

    rel2id = np.loadtxt(open(path + "relation2id.txt", "rb"), delimiter="\t", skiprows=1, dtype="str")
    rel2id_ = rel2id.copy()#np.zeros(shape=rel2id.shape, dtype=str)
    rel2id_[:,0]= rel2id[:,1]
    rel2id_[:,1]= rel2id[:,0]
    np.savetxt(output_path + "relations.dict",rel2id_,fmt='%s',delimiter="\t")


    train = np.loadtxt(open(path + "train2id.txt", "rb"), delimiter=" ",  skiprows=1, dtype="str")
    train_ = train.copy()#np.zeros(shape=train.shape, dtype=str)
    train_[:,0]= train[:,0]
    train_[:,2]= train[:,1]
    train_[:,1]= train[:,2]
    np.savetxt(output_path +"train.txt",train_,fmt='%s',delimiter="\t")

    test = np.loadtxt(open(path + "test2id.txt", "rb"), delimiter=" ",  skiprows=1, dtype="str")
    test_ = test.copy()#np.zeros(shape=test.shape, dtype=str)
    test_[:,0]= test[:,0]
    test_[:,2]= test[:,1]
    test_[:,1]= test[:,2]
    np.savetxt(output_path + "test.txt",test_,fmt='%s',delimiter="\t")

    valid = np.loadtxt(open(path + "valid2id.txt", "rb"), delimiter=" ", skiprows=1, dtype="str")
    valid_ = valid.copy()  # np.zeros(shape=test.shape, dtype=str)
    valid_[:, 0] = valid[:, 0]
    valid_[:, 2] = valid[:, 1]
    valid_[:, 1] = valid[:, 2]

    np.savetxt(output_path+ "valid.txt",valid_,fmt='%s',delimiter="\t")


input_path = "./input/"
output_path = "./output/"

folders = [ name for name in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, name)) ]

for folder in folders:
    covert_files_in_folder(input_path+ folder + "/", output_path+ folder.lower() + "/")