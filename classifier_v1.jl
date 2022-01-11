# classifier_v1.jl
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
classifier discussed in Chapter 1 of M. Nielsen's book:

 - Michael A. Nielsen, _Neural Networks and Deep Learning_, Determination Press (2015),
   http://neuralnetworksanddeeplearning.com/index.html

The network uses logistic neurons and learns through stochastic gradient
descent and a quadratic cost function.
"""
module Classifiers

import Random

export Classifier, feedforward, classify, train

"""
    struct Classifier

Datatype to represent a multiple layer network.  It stores the weights
and biases in arrays indexed by layer number.  The input layer is
layer 1, this has no biases or weights, but is stored as a 0 because
it makes numbering more convenient.
"""
struct Classifier
    nlayers::Int16
    size::Array{Int16,1}
    bias::Array{Vector{Float64},1}
    weight::Array{Matrix{Float64},1}
end

"""
    Classifier(layer_size::Vector{Int})

Construct a multiple layer network with layers of the given sizes and activation
function `σ`.  `σ_p` must be the derivative of `σ` or the learning methods will fail.
 Weights and biases are assigned randomly.

# Example
    net=Classifier([40, 20, 10])

Constructs a 3-layer network with 40 input neurons, 10 output neurons and a hidden layer
of 20 neurons.
"""
function Classifier(layer_size::Vector{Int})
    b=[ i==1 ? [0] : Random.rand(layer_size[i]).-.5 for i=1:length(layer_size) ]
    w=[ i==1 ? [0 0] : Random.rand(Float64,(layer_size[i],layer_size[i-1])).-.5 for i=1:length(layer_size) ]
    Classifier(length(layer_size),layer_size,b,w)
end

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
    train(net::Classifier, training_data, epochs, mini_batch_size, η, test_data=nothing)

Train the classifier `net` using mini-batch stochastic
gradient descent with a quadratic cost function.  `training_data` is a an array of tuples
`(x, y)` representing the training inputs and the desired outputs.  Both `x` and `y` must be
floating-point vectors matching the sizes of the input and output layers of `net`.  Note that
`y` is not simply the category value, but a vector with a 1 indicating the category.

If `test_data` is provided then the network will be evaluated against
the test data after each epoch, and partial progress printed out.
This is useful for tracking progress, but slows things down.
"""
function train(net::Classifier, training_data, epochs, mini_batch_size, η, test_data=nothing)
    tdata=copy(training_data)
    for j in 1:epochs
        Random.shuffle!(tdata)
        for k in 1:mini_batch_size:length(tdata)        
            update_mini_batch(net, tdata[k:min(length(tdata),k+mini_batch_size-1)], η)
        end
        if !isnothing(test_data)
            print("Epoch $j: $(evaluate(net,test_data)) / $(length(test_data))\n")
        else
            print("Epoch $j complete\n")
        end
    end
end

"""
    update_mini_batch(net::Classifier, mini_batch, η)

Update the network's weights and biases by applying gradient descent
using backpropagation to a single mini batch.  `mini_batch` is the same
format as `training_data` in function `train`, and `η` is the learning rate.
"""
function  update_mini_batch(net::Classifier, mini_batch, η)
    ∇_b = zero.(net.bias)
    ∇_w = zero.(net.weight)
    for (x, y) in mini_batch
        i_∇_b, i_∇_w = backprop(net, x, y)
        ∇_b .+= i_∇_b
        ∇_w .+= i_∇_w
    end
    n=length(mini_batch)
    net.weight .-= (η/n)*∇_w
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
    δ  = msqerr_derivative(a[L], y) .* σ_p.(z[L])
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
    msqerr_derivative(output_activations, y)

Return the vector of partial derivatives ∂C_x / ∂a of the quadritic cost function
with respec to the output activations
"""
msqerr_derivative(output_activations, y) = 0.5* (output_activations-y)

"""
    evaluate(net::Classifier, test_data)

Return the number of test inputs for which the neural
network outputs the correct result.
"""

function evaluate(net::Classifier, test_data)
    test_results = [ ( classify(net,x), y)
                        for (x, y) in test_data]
    return sum(Int(x == y) for (x, y) in test_results)
end

end
