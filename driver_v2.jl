#
# driver_v2.jl -- Driver to train the classifier implemented in classifier_v1.jl with
#                 MNIST digit data
#
# This follows closely the training procedure explained in chapter 1 of Nielsen's book,
# except that MNIST data is obtained using the MLDatasets Julia package
#
# This file copyright (C) 2022 by Tomas S. Grigera.
# 
# This is free software; you can redistribute it and/or modify it
# under the terms of the GNU General Public License (GPL) as
# published by the Free Software Foundation. You can use either
# version 3, or (at your option) any later version.
#
# This software is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
# or FITNESS FOR A PARTICULAR PURPOSE.
#
# For details see the file LICENSE in the home directory.
#

include("classifier_v2.jl")

import MLDatasets

#
# Read MNIST training and test data.  Training data is separated in
# train and validation data.
#
function mnist_data()
    ntrain=50000
    trainx,trainy = MLDatasets.MNIST.traindata()
    input_size=size(trainx,1)*size(trainx,2)
    trainx = Vector{Float32}[ reshape(trainx[:,:,i],input_size) for i=1:size(trainx,3) ]
    valid = [ (trainx[i],trainy[i]) for i=ntrain+1:size(trainx,1) ]

    trainy = [ Classifiers.vectorized(trainy[i]) for i=1:ntrain ]
    train = [ (trainx[i],trainy[i]) for i=1:ntrain ]

    testx,testy = MLDatasets.MNIST.testdata()
    testx = Vector{Float32}[ reshape(testx[:,:,i],input_size) for i=1:size(testx,3) ]
    test = [ (tx,ty) for (tx,ty) in zip(testx,testy) ]

    return train,valid,test
end

# Load data
train_d,valid_d,test_d=mnist_data()
input_size=length(train_d[1][1])

#
# Examples training with different cost functions.
# Hyperparameters should be tuned
#


# Create and train with quadratic cost, no regularisation
net1=Classifiers.Classifier([input_size, 30, 10], Classifiers.Quadratic_cost() );
Classifiers.train!(net1,train_d,30,10,0.5,λ=0,
                   evaluation_data=valid_d,
                   monitor_evaluation_accuracy=true,
                   monitor_evaluation_cost=true,
                   monitor_training_accuracy=true,
                   monitor_training_cost=true);

# Create and train with cross-entropy cost, no regularisation
net2=Classifiers.Classifier([input_size, 30, 10], Classifiers.CrossEntropy_cost() );
Classifiers.train!(net2,train_d,30,10,0.5,λ=0, evaluation_data=valid_d,monitor_evaluation_accuracy=true,
                   monitor_evaluation_cost=true, monitor_training_accuracy=true,
                   monitor_training_cost=true) ;

# Create and trains with quadratic cost and regularisation
net3=Classifiers.Classifier([input_size, 30, 10], Classifiers.Quadratic_cost() );
Classifiers.train!(net3,train_d,30,10,0.25,λ=0.005,
                   evaluation_data=valid_d,
                   monitor_evaluation_accuracy=true,
                   monitor_evaluation_cost=true,
                   monitor_training_accuracy=true,
                   monitor_training_cost=true);

# Create and train with cross-entropy cost and regularisation
net4=Classifiers.Classifier([input_size, 100, 10], Classifiers.CrossEntropy_cost() )
Classifiers.train!(net4,train_d,30,10,0.1,λ=0.001,
                   evaluation_data=valid_d,
                   monitor_evaluation_accuracy=true,
                   monitor_evaluation_cost=true,
                   monitor_training_accuracy=true,
                   monitor_training_cost=true);

print("end\n");
