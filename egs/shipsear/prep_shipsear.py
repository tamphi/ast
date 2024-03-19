import numpy as np
import json
import os
from sklearn.model_selection import KFold, train_test_split
from sklearn.utils import shuffle

meta = np.loadtxt('data/meta/ship_1000ms_chunks_44100k.csv', delimiter=',', dtype='str',skiprows=1)
base_dir = 'data/shipsEar_AUDIOS_chunk/'
total_chunks = 11368

label_set = np.loadtxt('./data/5_shipsear_class_labels_indices.csv', delimiter=',', dtype='str')
label_map = {}
for i in range(1, len(label_set)):
    label_map[eval(label_set[i][2])] = label_set[i][0]
print(label_map)

X_idx= np.array(range(0,total_chunks)) #index of audio 

#split into train, valid, test set
eval_size = 0.2
X_res,X_test = train_test_split(X_idx, shuffle=True,random_state=12,test_size=eval_size)
valid_size = 0.3
X_train,X_valid = train_test_split(X_res,shuffle= True, random_state = 56, test_size=valid_size)

# Assert that none of the sets have overlap datapoints 
assert len(np.intersect1d(X_train,X_valid)) == len(np.intersect1d(X_test,X_valid))  == len(np.intersect1d(X_test,X_train))  == 0

print("Eval set size: {} Validation set size: {}".format(eval_size,valid_size))

train_wav_list = []
val_wav_list = []
eval_wav_list = []

eval_dummy = []
val_dummy =[]
train_dummy = []
for i in X_train:
    label = label_map[meta[i][3]]
    train_dict = {"wav":base_dir + meta[i][1],"labels":'/m/07rwj'+label.zfill(2)}
    train_wav_list.append(train_dict)
    train_dummy.append(base_dir + meta[i][1])
for j in X_valid:
    label = label_map[meta[j][3]]
    val_dict = {"wav":base_dir + meta[j][1],"labels":'/m/07rwj'+label.zfill(2)}
    val_wav_list.append(val_dict)
    val_dummy.append(base_dir + meta[j][1])
for k in X_test:
    label = label_map[meta[k][3]]
    eval_dict = {"wav":base_dir + meta[k][1],"labels":'/m/07rwj'+label.zfill(2)}
    eval_wav_list.append(eval_dict)
    eval_dummy.append(base_dir + meta[k][1])

# overlap ='data/shipsEar_AUDIOS_chunk/67__23_07_13_H3_PirataCiesSale_chunk_75.wav'
# print(train_dummy.index(overlap))
# print(eval_dummy.index(overlap))
# print(val_dummy.index(overlap))
# print(np.intersect1d(train_dummy,val_dummy))
# print(np.intersect1d(train_dummy,eval_dummy))
# print(np.intersect1d(val_dummy,eval_dummy))

assert len(np.intersect1d(train_dummy,val_dummy)) == len(np.intersect1d(val_dummy,eval_dummy))  == len(np.intersect1d(eval_dummy,train_dummy))  == 0

##########Uncomment to save json 
# print('{:d} training samples,{:d} validation samples, {:d} test samples'.format(len(train_wav_list), len(val_wav_list),len(eval_wav_list)))
    
#     #save to json file
# with open('data/5_datafiles/shipsear_train_data.json', 'w') as f:
#     json.dump({'data': train_wav_list}, f, indent=1)

# with open('data/5_datafiles/shipsear_valid_data.json', 'w') as f:
#     json.dump({'data': val_wav_list}, f, indent=1)

# with open('data/5_datafiles/shipsear_eval_data.json', 'w') as f:
#     json.dump({'data': eval_wav_list}, f, indent=1)
#########

####Training with Kfold options
# X_idx = np.char.mod('%d', X_idx)
# X = shuffle(X_idx, random_state=42)

# kf = KFold(n_splits=5,random_state=56, shuffle=True)
# split_X = kf.split(X)

# for fold,(train_idx, test_idx) in enumerate(split_X):
#     train_wav_list = []
#     eval_wav_list = []
#     print(fold)
#     print(train_idx)

#     for idx in train_idx:
#         label = label_map[meta[idx][3]]
#         train_dict = {"wav":base_dir + meta[idx][1],"labels":'/m/07rwj'+label.zfill(2)}
#         train_wav_list.append(train_dict)
#     for idx in test_idx:
#         label = label_map[meta[idx][3]]
#         eval_dict = {"wav":base_dir + meta[idx][1],"labels":'/m/07rwj'+label.zfill(2)}
#         eval_wav_list.append(eval_dict)
    
#     print('fold {:d}: {:d} training samples, {:d} test samples'.format(fold, len(train_wav_list), len(eval_wav_list)))
    
#     #save to json file
#     with open('./data/datafiles/shipsear_train_data_'+ str(fold+1) +'.json', 'w') as f:
#         json.dump({'data': train_wav_list}, f, indent=1)

#     with open('./data/datafiles/shipsear_eval_data_'+ str(fold+1) +'.json', 'w') as f:
#         json.dump({'data': eval_wav_list}, f, indent=1)


print("Finished preparing ShipsEar dataset") 
    
    

