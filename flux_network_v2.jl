#
# flux_network_v2.jl -- Driver as driver_v2 but built on Flux
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

using Flux
import MLDatasets

"""
    vectorized(digit::Int)

Convert the digit classification to a 10-component vector
"""
function vectorized(digit::Int)
    v=zeros(typeof(digit),10)
    v[digit+1]=1
    return v
end

"""
     mnist_data()

Read MNIST training and test data.  Training data is separated in
train and validation data.  Data is returned as
`(train, validation, test)` tuple.  Each element is in turn a tuple
`(x,y)`.  For convenience, `y` is a digit in the case of validation and
test data, and a 10-element vector (obtained through `vectorize`) in
the case of training data.
"""
function mnist_data()
    ntrain=50000
    trainx,trainy = MLDatasets.MNIST.traindata()

    input_size=size(trainx,1)*size(trainx,2)
    trainx = reshape(trainx[:,:,1:ntrain],input_size,ntrain)
    trainy = hcat(vectorized.(trainy[1:ntrain])...)
    train = (trainx,trainy)

    valid = ( reshape(trainx[:,:,ntrain+1:end],input_size,:), trainy[ntrain+1:end] )

    testx,testy = MLDatasets.MNIST.testdata()
    test = ( reshape(testx,input_size,:), testy ) 

    return train,valid,test
end

"""
    train!(net, training_data, cost, optimizer;
           epochs=1, batch_size=10, 
           evaluation_data=nothing, monitor_evaluation_cost=false,
           monitor_evaluation_accuracy=false,
           monitor_training_cost=false, monitor_training_accuracy=false)

Train the network `net` using cost function `cost` and the given
`optimizer`. `training_data` is a  tuple `(x, y)` of matrices representing the
training inputs and the desired outputs.  The columns of `x` and `y` must be
floating-point vectors matching the sizes of the input and output layers of `net`.

Trainig will proceed for the requested number of epochs, and the gradient of
the cost function will be computed over mini-batches of the training data
of length `batch_size`.

The network will be evaluated for cost and accuracy with the training and optional
evaluation data accordiing to the last four flags.
"""
function train!(net, training_data, cost, optimizer;
                epochs=1, batch_size=10, 
                evaluation_data=nothing, monitor_evaluation_cost=false,
                monitor_evaluation_accuracy=false,
                monitor_training_cost=false, monitor_training_accuracy=false)

    evaluation_cost, evaluation_accuracy = (Float64)[], (Float64)[]
    training_cost, training_accuracy = (Float64)[], (Float64)[]
    if monitor_evaluation_cost || monitor_evaluation_accuracy
        ex, ey = evaluation_data
        evaluation_data_v = (ex, hcat(vectorized.(ey)...) )
    end
    parameters = params(net)
    dpoints=size(training_data[1],2)
    if !isnothing(evaluation_data)
        epoints=size(evaluation_data[1],2)
    end

    for e=1:epochs
        for k in 1:batch_size:dpoints
            brange=k:min(dpoints,k+batch_size-1)
            x = training_data[1][:,brange]
            y = training_data[2][:,brange]
            gs = gradient(() -> cost(x, y), parameters)
            Flux.Optimise.update!(opt, parameters, gs)
        end

        print("Epoch $e complete\n")
        if monitor_training_cost
            c=cost(training_data...)
            push!(training_cost,c)
            print("-- Cost on training data: $c\n")
        end
        if monitor_training_accuracy
            acc = accuracy(net,training_data,true)
            push!(training_accuracy,acc)
            n=length(training_data)
            print("-- Accuracy on training data: $acc / $dpoints ($(acc/dpoints))\n")
        end
        if monitor_evaluation_cost
            c =cost(evaluation_data_v...)
            push!(evaluation_cost,c)
            print("-- Cost on evaluation data: $c\n")
        end
        if monitor_evaluation_accuracy
            acc = accuracy(net,evaluation_data)
            push!(evaluation_accuracy,acc)
            n=length(evaluation_data)
            print("-- Accuracy on evaluation data: $acc / $epoints ($(acc/epoints))\n")
        end
    end
    return (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)
end

function accuracy(net, data, convert=false)
    rdigits = map(x->x[1],findmax(net(data[1]),dims=1)[2]).-1
    if convert
        yd = map(x->x[1],findmax(data[2],dims=1)[2]).-1 
   else
        yd=data[2]
    end
    return sum(Int(x == y) for (x, y) in zip(rdigits,yd) )
end


# Load data
train_d,valid_d,test_d=mnist_data()
input_size=size(train_d[1],1)

# Create and train with quadratic cost, no regularisation
# net = Chain(Dense(input_size,30,NNlib.sigmoid,init=Flux.glorot_normal),
#             Dense(30,10,NNlib.sigmoid,init=Flux.glorot_normal))
# cost(x, y) = Flux.Losses.mse(net(x), y)
# opt = Descent(2.)
# train!(net,train_d,cost,opt,epochs=30,batch_size=10,evaluation_data=test_d,
#        monitor_evaluation_cost=true,monitor_evaluation_accuracy=true,
#        monitor_training_cost=true,monitor_training_accuracy=true)

# # Create and train with cross-entropy cost, no regularisation
# net = Chain(Dense(input_size,30,NNlib.sigmoid,init=Flux.glorot_normal),
#             Dense(30,10,NNlib.sigmoid,init=Flux.glorot_normal))
# opt = Descent(2.)
# cost(x, y) = Flux.Losses.binarycrossentropy(net(x), y)
# train!(net,train_d,cost,opt,epochs=30,batch_size=10,evaluation_data=test_d,
#        monitor_evaluation_cost=true,monitor_evaluation_accuracy=true,
#        monitor_training_cost=true,monitor_training_accuracy=true)

# # Create and train with quadratic cost and regularisation
# net = Chain(Dense(input_size,30,NNlib.sigmoid,init=Flux.glorot_normal),
#             Dense(30,10,NNlib.sigmoid,init=Flux.glorot_normal))
# mini_batch_size=30
# λ=1e-3/mini_batch_size
# cost(x, y) = Flux.Losses.mse(net(x), y) + λ*sum(x->sum(y->y^2,x),params(net))
# opt = Descent(2.)
# train!(net,train_d,cost,opt,epochs=30,batch_size=mini_batch_size,evaluation_data=test_d,
#        monitor_evaluation_cost=true,monitor_evaluation_accuracy=true,
#        monitor_training_cost=true,monitor_training_accuracy=true)

# Create and train with cross-entropy cost and regularisation
hidden_size=100
net = Chain(Dense(input_size,hidden_size,NNlib.sigmoid,init=Flux.glorot_normal),
            Dense(hidden_size,10,NNlib.sigmoid,init=Flux.glorot_normal))
mini_batch_size=30
opt = Descent(15.)
λ=1e-3/mini_batch_size
cost(x, y) = Flux.Losses.binarycrossentropy(net(x), y) + λ*sum(x->sum(y->y^2,x),params(net))
train!(net,train_d,cost,opt,epochs=30,batch_size=mini_batch_size,evaluation_data=test_d,
       monitor_evaluation_cost=true,monitor_evaluation_accuracy=true,
       monitor_training_cost=true,monitor_training_accuracy=true)

print("end\n")
