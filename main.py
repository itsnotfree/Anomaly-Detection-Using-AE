# Anomaly Detection Using Auto-Encoder
# Omri Rafaeli

import numpy as np
from autoEncoder import Autoencoder

#Parameters Definition
epochs = 100
batchSize=50
Training_Lenght = 200
num_of_features = 8
encoding_dim = 4
training_precentage=0.95
testing_length=100


# Training Inputs Creation (Non-Anomaly - random [0,1] vector)
input_mat = np.zeros((num_of_features, Training_Lenght))
for col in range(Training_Lenght):
    input_mat[:,col] = np.random.rand(1,num_of_features)

input_mat=np.transpose(input_mat)
num_of_samples,Input_dimension = input_mat.shape      # can be taken also from parameters definition
(x_train, x_test) = np.split(input_mat, [int(num_of_samples * training_precentage)])

#AE creation
autoenc = Autoencoder(Input_dimension, encoding_dim)
#printing ae summary
print (autoenc.summary())
#training ae
autoenc.train(x_train,x_test,epochs,batchSize)

#find maximum mse (in order to find Thr)
max_mse=0
sample = np.zeros((num_of_features,1))
for i in range(num_of_samples):
    sample=np.transpose(sample)
    sample = (input_mat[i,:]).reshape(1,num_of_features)
    decoded_sample = autoenc.prediction(sample)
    sample_mse=np.square(np.subtract(sample,decoded_sample)).mean()
    if (sample_mse>max_mse):
        max_mse=sample_mse

print("Max MSE out of Non Anomaly Inputs is " + str(max_mse))

# Creat test dataset
testing_mat = np.zeros((num_of_features, testing_length))
for col in range(testing_length):
    if(col%2==0):
        testing_mat[:,col] = np.random.rand(1,num_of_features)    # Non Anomaly - [0,1] with mean 0.5
    else:
        testing_mat[:,col] = 2*np.random.rand(1, num_of_features) #Anomaly -  [0,2] with mean 1

# inserting unseen inputs to the Anomaly detection unit
anomaly_bin_vec=[]
sample = np.zeros((num_of_features,1))
for i in range(testing_length):
    sample=np.transpose(sample)
    sample = (testing_mat[:,i]).reshape(1,num_of_features)
    decoded_sample = autoenc.prediction(sample)
    sample_mse=np.square(np.subtract(sample,decoded_sample)).mean()
    if (sample_mse>max_mse):
        anomaly_bin_vec.append(1)
    else:
        anomaly_bin_vec.append(0)


#Detection Statistics
Detection_Precentage = sum((k==1 and anomaly_bin_vec.index(k)%2==1) for k in anomaly_bin_vec)/(0.5*testing_length)
print("Detection Precentage is - "+ str(Detection_Precentage))