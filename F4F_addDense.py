import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
os.environ["TF_CPP_MIN_LOG_LEVEL"] = '3'
import tensorflow as tf
import numpy as np
import PIL
import gc
gc.collect()
#import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras import Sequential
from tensorflow.compat.v1 import ConfigProto
from tensorflow.python.client import device_lib
print("GPU? :",tf.config.list_physical_devices('GPU'))
model_path = "/media/2/Network/pretrained_model/vgg_model.h5"
#model_path = "/media/2/Network/pretrained_model/back_layers.h5"
#img_path = "/media/2/Network/Imagenet_dup/val/n02074367"# dugong
data_path = "/media/2/Network/Imagenet_dup/"
# 16 error
feature4_path = "/media/2/Network/extracted_feature/whole_not_shuffle_to_15/seq_16_pkt_error"
# 16 error case
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.4
config.gpu_options.allow_growth = True

model=None
def ready_model(model_path,layer):
    global til_pooling4_model,til_pooling4_predict, til_pooling5_model,til_pooling5_predict, model
    model=load_model(model_path) # whole model
    # pooling 4
    if layer in 'pooling4':
        print('pooling4 session')
        #til_pooling4_model = Sequential([layer for layer in model.layers[:15]]) 
        #til_pooling4_model.build((None, 224,224,3))
        til_pooling4_predict = Sequential([layer for layer in model.layers[15:]]) 
        til_pooling4_predict.build((None, 14,14,512))
    # pooling5
    elif layer in 'pooling5':
        print('pooling5 session')
        til_pooling5_model = Sequential([layer for layer in model.layers[:19]]) 
        til_pooling5_model.build((None, 224,224,3))
        til_pooling5_predict = Sequential([layer for layer in model.layers[19:]]) 
        til_pooling5_predict.build((None, 7,7,512))
# load model & split pooling4 & pooling5
#til_pooling4_model=None
til_pooling4_predict=None
'''
til_pooling5_model=None
til_pooling5_predict=None
'''
ready_model(model_path,'pooling4')

feature_list =  os.listdir(feature4_path)
feature_list = sorted(feature_list)
len(feature_list)

# classify feature
test_feature = []
test_label = []
train_feature = []
train_label =[]
val_feature = []
val_label =[]
for data in feature_list:
    #print(feature4_path+"/"+data)
    if 'train' in data:
        if 'feature' in data:
            train_feature.append(np.load(feature4_path+"/"+data,mmap_mode='c'))
        else :
            train_label.append(np.load(feature4_path+"/"+data,mmap_mode='c'))
       #print("train in",data)
    elif 'test' in data:
        #print("test in",data)
        if 'feature' in data:
            test_feature.append(np.load(feature4_path+"/"+data,mmap_mode='c'))
        else:
            test_label.append(np.load(feature4_path+"/"+data,mmap_mode='c'))
    elif 'validation' in data:
        #print("val in",data)
        if 'feature' in data:
            val_feature.append(np.load(feature4_path+"/"+data,mmap_mode='c'))
        else:
            val_label.append(np.load(feature4_path+"/"+data,mmap_mode='c'))
            
            
# add error in feature
def error_injection(feature,num_of_error):
    num_ch = feature.shape[3]
    for i, data in enumerate(feature):
        #print(i,data.shape)
        start = (i*num_of_error) % num_ch
        end = ((i+1)*num_of_error) % num_ch
        data[:,:,start:end] = 0
# test code
'''
tmp = np.zeros((2,2,2,8))
tmp.shape # (14, 14, 2)
error_injection(tmp,num_of_error)    
'''  

    
back_layer = tf.keras.Sequential()
back_layer.add(tf.keras.layers.Flatten(name='flatten1'))
gc.collect()
#fc_layer = tf.keras.layers.Dense(14*14*512,input_dim=14*14*512,activation='tanh')
fc_layer1 = tf.keras.layers.Dense(512,input_dim=1,activation='tanh')
back_layer.add(fc_layer1)
fc_layer2 = tf.keras.layers.Dense(14*14*512,input_dim=1,activation='tanh')
back_layer.add(fc_layer2)
back_layer.add(tf.keras.layers.Reshape((14,14,512)))
for layer in til_pooling4_predict.layers[:]: # til_pooling4_predict.layers[1:]
    layer.trainable = False
    back_layer.add(layer)
gc.collect()
#back_layer = tf.keras.layers.Concatenate()(fc_layer,til_pooling4_predict)
# model  compile
def scheduler(epoch,lr):
    if epoch < 15:
        return lr
    else:
        return lr*tf.math.exp(-0.1)
MODEL_SAVE_FOLDER_PATH = './models/'
if not os.path.exists(MODEL_SAVE_FOLDER_PATH):
    os.mkdir(MODEL_SAVE_FOLDER_PATH)
model_path = MODEL_SAVE_FOLDER_PATH + 'epoch_{epoch:02d}-val_loss_{val_loss:.4f}.hdf5'
cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=model_path, 
                               monitor='val_loss', verbose=1,
                               save_best_only=True)

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(scheduler)
optimizer = tf.keras.optimizers.Adam(lr=1e-4, decay=0)
#til_pooling4_predict.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
back_layer.compile(optimizer=optimizer, loss='categorical_crossentropy',metrics=['accuracy'])
back_layer.build((None,14,14,512))
back_layer.summary()

max_epoch = 60
feature_len = len(train_feature[0])
val_feature_list_len = len(val_feature)
#for epoch in range(1,max_epoch+1):
#    print("epoch :",epoch)
print(feature_len, val_feature_list_len)
for epoch in range(1,max_epoch+1):
    for i, feature in enumerate(train_feature):
     #   print(feature.shape,val_feature[i].shape)
        print("== epoch %2d == training phase -> %d th feature" % (epoch, i)) 
     #   break # debug
        val_range = i % val_feature_list_len
        start = (val_range*feature_len)
        end  = ((val_range+1)*feature_len) 
        if i is 62: # 9169
            y = back_layer.fit(feature,train_label[0][-9169:],
                        batch_size=32,epochs=3,
                             validation_data=(val_feature[i%val_feature_list_len],val_label[0][start:end]),
                                   #callbacks=[cb_checkpoint],verbose=1,use_multiprocessing=False)
                                   verbose=1,use_multiprocessing=False)
        
        else:
            y = back_layer.fit(feature,train_label[0][start:end],
                        batch_size=32,epochs=3,
                             validation_data=(val_feature[i%val_feature_list_len],val_label[0][start:end]),
                                   #callbacks=[cb_checkpoint],verbose=1,use_multiprocessing=False)
                                   verbose=1,use_multiprocessing=False)
    if epoch >=30:
        min_loss =10000
        max_acc =0
        for i in range(len(test_feature)):
            print("== evaluate mode == at epoch %d" % epoch)
            start = i*feature_len
            end = (i+1)*feature_len
#            print(test_feature[i].shape, test_label[0][start:end].shape)
            loss,acc = back_layer.evaluate(test_feature[i],test_label[0][start:end])
            print(epoch,"th section loss : %.2f" %loss)
            print(epoch,"th section acc : %.2f %%" %(acc))
            if min_loss > loss :
                min_loss = loss
            if max_acc < acc:
                max_acc = acc
        #    break # debug
        if epoch % 5 is 0:
            if max_acc is 0:
                print("ERROR : epoch(%d)max_acc is zero" % (epoch))
                break
            back_layer.save('./models/epoch_'+str(epoch)+"loss_%.2facc_%.2f_h5" % (min_loss,max_acc))
    
