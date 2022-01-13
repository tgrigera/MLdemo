# classifier_v2.jl
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


"""
    module Classifiers

This implements a multiple-layer neural network to be used as a
classifier.  It is essentially a Julia implementation of the
classifier discussed in Chapter 3 of M. Nielsen's book:

 - Michael A. Nielsen, _Neural Networks and Deep Learning_, Determination Press (2015),
   http://neuralnetworksanddeeplearning.com/index.html

It is a more flexible version of classifier_v1.jl, including different cost functions
and optional regularization.
"""
module Classifiers

import LinearAlgebra
import Random
import Distributions

export Classifier, feedforward, classify, train!
export Quadratic_cost,CrossEntropy_cost

#
# Cost functions
#

abstract type Cost_function end

struct CrossEntropy_cost <: Cost_function
end

cost_fn(c::CrossEntropy_cost, a, y) = 
    sum(replace!( -y.*log.(a)-(1 .-y).*log.(1 .-a), -Inf=>0) )

cost_delta(c::CrossEntropy_cost,z, a, y) = a.-y

struct Quadratic_cost <: Cost_function
end

cost_fn(c::Quadratic_cost,a, y) = LinearAlgebra.norm(a-y)^2

cost_delta(c::Quadratic_cost,z, a, y) = 2 * (a.-y) .* σ_p.(z)


"""
    struct Classifier

Datatype to represent a multiple layer network.  It stores the weights
and biases in arrays indexed by layer number.  The input layer is
layer 1, this has no biases or weights, but is stored as a 0 because
it makes numbering more convenient.

It also stores pointer to
"""
struct Classifier{costT<:Cost_function}
    nlayers::Int16
    size::Array{Int16,1}
    bias::Array{Vector{Float64},1}
    weight::Array{Matrix{Float64},1}
    cost::costT
end

"""
    Classifier(layer_size::Vector{Int},cost::Cost_function)

Construct a multiple layer network with layers of the given sizes and cost function.
Weights and biases are assigned from a Gaussian distribution with 0 mean and variance 1/N
and 1 respectively.

# Example
    net=Classifier([40, 20, 10])

Constructs a 3-layer network with 40 input neurons, 10 output neurons and a hidden layer
of 20 neurons.
"""
function Classifier(layer_size::Vector{Int},cost::costT) where {costT<:Cost_function}
    gauss = Distributions.Normal(0,1)
    b=[ i==1 ? [0.] : Random.rand(gauss,layer_size[i]) for i=1:length(layer_size) ]
    w=[ i==1 ? [0. 0.] : Random.rand(gauss,(layer_size[i],layer_size[i-1]))./sqrt(layer_size[i-1])
        for i=1:length(layer_size) ]
    Classifier(Int16(length(layer_size)),Int16.(layer_size),b,w,cost)
end

#
# Activation function and feed forward
#

"""
    σ(x)

Evaluate the logistic/sigmoid function 1/(1+exp(-x))
"""
σ(z) = 1.0/(1.0+exp(-z))

"""
    σ_p(x)

Evaluate the derivative of the sigmoid function `σ(x)`.
"""
σ_p(z) = σ(z)*(1-σ(z))


"""
    feedforward(net::Classifier,x)

Return the output of the classifier `net` for the input layer values `x`
"""
function feedforward(net::Classifier,x)
    a=copy(x)
    for l in 2:net.nlayers
         a=σ.(net.weight[l]*a+net.bias[l])
    end
    return a
end

"""
    classify(net::Classifier,x)

Find the category assigned to `x` by the classifier `net`.  Output is an integer
in the range 0:size(output layer)-1
"""
classify(net::Classifier,x) = findmax(feedforward(net,x))[2]-1

"""
    train!(net::Classifier, training_data, epochs, mini_batch_size, η;
           λ=0, evaluation_data=nothing, monitor_evaluation_cost=false,
           monitor_evaluation_accuracy=false, monitor_training_cost=false,
           monitor_training_accuracy=false)


Train the classifier `net` using mini-batch stochastic
gradient descent with the cost function chosen when creating `net`. 
`training_data` is a an array of tuples `(x, y)` representing the
training inputs and the desired outputs.  Both `x` and `y` must be
floating-point vectors matching the sizes of the input and output layers of `net`.
Note that `y` is not simply the category value, but a vector with a 1
indicating the category (which can be obtained with the `vectorized` function).

`λ` is the parameter for L2 regularisation.

The network will be evaluated for cost and accuracy with the training and optional
evaluation data accordiing to the lasr four flags.
"""
function train!(net::Classifier, training_data, epochs, mini_batch_size, η;
                λ=0, evaluation_data=nothing, monitor_evaluation_cost=false,
                monitor_evaluation_accuracy=false, monitor_training_cost=false,
                monitor_training_accuracy=false)

    tdata=copy(training_data)
    evaluation_cost, evaluation_accuracy = (Float64)[], (Float64)[]
    training_cost, training_accuracy = (Float64)[], (Float64)[]

    for j in 1:epochs
        Random.shuffle!(tdata)
        for k in 1:mini_batch_size:length(tdata)        
            update_mini_batch!(net, tdata[k:min(length(tdata),k+mini_batch_size-1)], η, λ)
        end
        print("Epoch $j complete\n")
        if monitor_training_cost
            cost = total_cost(net, training_data, λ)
            push!(training_cost,cost)
            print("— Cost on training data: $cost\n")
        end
        if monitor_training_accuracy
            acc = accuracy(net, training_data,true)
            push!(training_accuracy,acc)
            n=length(training_data)
            print("— Accuracy on training data: $acc / $n ($(acc/n))\n")
        end
        if monitor_evaluation_cost
            cost = total_cost(net,evaluation_data, λ, true)
            push!(evaluation_cost,cost)
            print("— Cost on evaluation data: $cost\n")
        end
        if monitor_evaluation_accuracy
            acc = accuracy(net,evaluation_data)
            push!(evaluation_accuracy,acc)
            n=length(evaluation_data)
            print("— Accuracy on evaluation data: $acc / $n ($(acc/n))\n")
        end
    end
    return (evaluation_cost, evaluation_accuracy, training_cost, training_accuracy)
end

"""
    update_mini_batch!(net::Classifier, mini_batch, η, λ)

Update the network's weights and biases by applying gradient descent
using backpropagation to a single mini batch.  `mini_batch` is the same
format as `training_data` in function `train`, `η` is the learning rate,
and `λ` is the regularization parameter.
"""
function update_mini_batch!(net::Classifier, mini_batch, η, λ)
    ∇_b = zero.(net.bias)
    ∇_w = zero.(net.weight)
    for (x, y) in mini_batch
        i_∇_b, i_∇_w = backprop(net, x, y)
        ∇_b .+= i_∇_b
        ∇_w .+= i_∇_w
    end
    n=length(mini_batch)
    net.weight .-= η*λ/n * net.weight .+ (η/n)*∇_w
    net.bias .-= (η/n)*∇_b
end

"""
    backprop(net::Classifier, x, y)

Return a tuple `(∇_b, ∇_w)` representing the gradient for the
quadratic cost function for a single input.  `∇_b` and
`∇_w`` have the same structure used to represent biases and weights in
`struct Classifier`.
"""
function backprop(net::Classifier, x, y)
    ∇_b = zero.(net.bias)
    ∇_w = zero.(net.weight)
    L = net.nlayers

    # forward pass
    a = zero.(net.bias)  # store all activations and zs
    z = zero.(net.bias)
    a[1]=x
    for l=2:L
        z[l] = net.weight[l]*a[l-1] + net.bias[l]
        a[l] = σ.(z[l])
    end

    # backward pass
    δ  = cost_delta(net.cost, z[L], a[L], y)
    ∇_b[L] = δ
    ∇_w[L] = δ * a[L-1]'
    for l=L-1:-1:2
        δ  = net.weight[l+1]'*δ .* σ_p.(z[l])
        ∇_b[l] = δ
        ∇_w[l] = δ * a[l-1]'
    end
    return (∇_b, ∇_w)
end

"""
    total_cost(net::Classifier, data, λ, convert=false)

Return the total cost for data set `data`.  The flag
`convert` should be set to `false` if the data set is the training
data (the usual case), and to `true` if the data set is the validation
or test data.  See comments on the similar (but reversed) convention
for the `accuracy` method.
"""
function total_cost(net::Classifier, data, λ, convert=false)
    cost = 0.0
    for (x,y) in data
        a = feedforward(net,x)
        if convert  y = vectorized(y) end
        cost += cost_fn(net.cost,a, y)/length(data)
    end
    cost += 0.5*(λ/length(data))*sum(
        LinearAlgebra.norm(w)^2 for w in net.weight)
    return cost
end

"""
    accuracy(net::Classifier, data, convert=false)

Return the number of inputs in `data` for which the neural network
outputs the correct result. The neural network's output is assumed to
be the index of whichever neuron in the final layer has the highest
activation.

The flag `convert` should be set to `false` if the data set is
validation or test data (the usual case), and to True if the data set
is the training data. The need for this flag arises due to differences
in the way the results `y` are represented in the different data sets.
In particular, it flags whether we need to convert between the
different representations.  The different representations for the
different data sets are used for efficiency reasons — the program
usually evaluates the cost on the training data and the accuracy on
other data sets.  Using different representations speeds things up.
"""
function accuracy(net::Classifier, data, convert=false)
    if convert
        test_results = [ ( classify(net,x), findmax(y)[2]-1 )
                         for (x, y) in data]
    else
        test_results = [ ( classify(net,x), y)
                         for (x, y) in data]
    end
    return sum(Int(x == y) for (x, y) in test_results)
end


"""
    vectorized(digit::Int)

Convert the digit classification to a 10-component vector
"""
function vectorized(digit::Int)
    v=zeros(typeof(digit),10)
    v[digit+1]=1
    return v
end


end
