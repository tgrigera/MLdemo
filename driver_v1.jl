#
# driver_v1.jl -- Driver to train the classifier implemented in classifier_v1.jl with
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

include("classifier_v1.jl")

import MLDatasets

#
# Convert the digit classification to a 10-component vector
#
function vectorized(digit::Int)
    v=zeros(typeof(digit),10)
    v[digit+1]=1
    return v
end

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

    trainy = [ vectorized(trainy[i]) for i=1:ntrain ]
    train = [ (trainx[i],trainy[i]) for i=1:ntrain ]

    testx,testy = MLDatasets.MNIST.testdata()
    testx = Vector{Float32}[ reshape(testx[:,:,i],input_size) for i=1:size(testx,3) ]
    test = [ (tx,ty) for (tx,ty) in zip(testx,testy) ]

    return train,valid,test
end

#
# Train the network
#

train_d,valid_d,test_d=mnist_data()
input_size=length(train_d[1][1])
net=Classifiers.Classifier([input_size, 30, 10])
η=3.0
minib_size=10
nepochs=30
Classifiers.train(net,train_d,nepochs,minib_size,η,test_d)

# Now the network is trained, we try it on a few images.  This also
# shows the original image using ImageView, if you prefer not to install
# this package, just comment the final three lines of the for-loop

import Random
import ImageView

print("Network trained, testing on a few images:\n")
Nt=10
for i=1:Nt
    k=Random.rand(1:size(test_d,1))
    print("Test image $k: label=$(test_d[k][2]), classified as $(Classifiers.classify(net,test_d[k][1]))\n")
    img=MLDatasets.MNIST.convert2image(test_d[k][1])
    ImageView.imshow(1 .-img)
    print("Press enter for next image\n"); read(stdin,1);
    ImageView.closeall()
end
