import tensorflow as tf
import numpy as np
import os
import random
from tensorflow import keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras import datasets

#pre-process_input in batches
def pre_process_input(be,img_dir,img_size,model_name):
    images = np.zeros((be, img_size, img_size, 3))
    for i, img in enumerate(os.listdir(img_dir)):
        load_image = image.load_img(os.path.join(img_dir, img))
        array_image = image.img_to_array(load_image.resize((img_size, img_size)))
        images[i, :] = array_image
        if i == be-1:
            break
    if model_name == "resnet50":
        batch_holder = tf.keras.applications.resnet50.preprocess_input(images.copy())
    elif model_name == 'vgg16':
        batch_holder = tf.keras.applications.vgg16.preprocess_input(images.copy())
    else:
        print("Incorrect model_name")
    return batch_holder,images

def pre_process_cifar_input(no_of_images):
    (X_train, y_train), (X_test,y_test) = datasets.cifar10.load_data()
    images = X_test[0:len(X_test):int(len(X_test)/no_of_images)]
    random.seed(45)
    random.shuffle(X_test)
    images = X_test[0:no_of_images]
    batch_holder = images / 255.0
    return batch_holder,images

#class probabilites prediciton
def class_prob(img,model,model_name,class_id = None):
    if class_id is None:
        class_predictions = model.predict(img)
        class_id = np.argmax(class_predictions,axis=1)
        class_probal = np.max(class_predictions,axis=1)
        return class_id,np.round(class_probal*100,2)
    else:
        if model_name == 'resnet50':
            preprocessed = tf.keras.applications.resnet50.preprocess_input(img.copy())
        elif model_name == 'vgg16':
            preprocessed = tf.keras.applications.vgg16.preprocess_input(img.copy())
        elif model_name == 'custom_model':
            preprocessed = img / 255.0
        else:
            print("Incorrect model_name for Explanation map")
        class_predictions = model.predict(preprocessed)
        class_probal = np.zeros(len(class_id))
        for i in range(len(class_id)):
            class_probal[i] = class_predictions[i,class_id[i]]
        return np.round(class_probal*100,2)

# evaluating metrics
def evaluate_metrics(cam_batch,batch_holder,img,model,so_ba,batch_size,sum_i, count_Ioc, sum_un, count_Doc,model_name):
    E_c = cam_batch * img
    class_id,f_I = class_prob(batch_holder,model,model_name)
    f_E = class_prob(E_c, model,model_name,class_id)
    E_c_un = (1 - cam_batch) * img
    f_E_un = class_prob(E_c_un, model,model_name,class_id)
    for i in range(len(class_id)):
        if f_I[i] > f_E[i]:
            y_c = round(((abs(f_I[i] - f_E[i]) / f_I[i]) * 100), 2)
            sum_i += y_c


        # Increase of Confidence (IoC)
        if f_E[i] > f_I[i]:
            count_Ioc += 1

        # Increased Average Drop(IAD)
        if f_I[i] > f_E_un[i]:
            y_c_un = round(((abs(f_I[i] - f_E_un[i]) / f_I[i]) * 100), 2)
            sum_un += y_c_un

            # Decrease of Confidence (DoC)
        if f_E_un[i] > f_I[i]:
            count_Doc += 1
    so_ba += batch_size
    return sum_i, count_Ioc, sum_un, count_Doc,so_ba


# averaging the metrics and final result
def final_metrics(sum_i,count_Ioc,sum_un,count_Doc,No_of_images):
    AD = round(sum_i/No_of_images,2)
    print("Average Drop(AD): "+str(AD))
    Ioc = (count_Ioc/No_of_images)*100
    print("Increase of confidence(Ioc): "+str(Ioc))
    IAD = round(sum_un/No_of_images,2)
    print("Increased Average Drop(IAD): "+str(IAD))
    Doc = (count_Doc/No_of_images)*100
    print("Decrease of Confidence(Doc): "+str(Doc))