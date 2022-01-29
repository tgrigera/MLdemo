# flux_test.jl --- Simple test of the flux package: curve fitting
#
# NB: This code is intended for playing around with Flux and fully-connected
# networks, it is not intended to imply that this is the best way to do
# nonlinear curve-fitting.
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
using Random
using GLMakie
GLMakie.activate!()

Random.seed!(3321)

# This is a test/demo of the flux package, used to fit 1-d data

function gendata(func,interval)
    x=collect(interval)
    x=collect(x')
    y=func.(x)*1.0
    y .+= y[3]*Random.rand(length(x))'
    return (x,y)
end

function mytrain!(cost,parameters,data,opt)
    for minib in data
        x, y = minib
        gs = gradient(() -> cost(x, y), parameters)
        Flux.Optimise.update!(opt, parameters, gs)
    end
end

#
# First test: linear regression
#
linear(x) = 5x+10
trainx,trainy=gendata(linear,0:10)
data=[(trainx,trainy)]
#data=collect(zip(trainx,trainy))

model = Dense(1, 1)
qcost(x, y) = Flux.Losses.mse(model(x), y)
opt = Descent(0.1/length(trainx))    # It seems it is better to divide by the length of the (mini)batch
parameters = params(model)
print("Initial cost: $(qcost(trainx,trainy)), parameters $parameters\n")
for i=1:1000
    mytrain!(qcost, parameters, data, opt)
end
print("Final cost: $(qcost(trainx,trainy)), parameters $parameters\n")

fig=Figure();
ax=fig[1,1]=Axis(fig,title="Linear fit, y=5x+10 ($(sum(length.(parameters))) parameters)")
scatter!(ax,vec(trainx),vec(trainy))
lines!(ax,vec(trainx),vec(model(trainx)))
display(fig)
print("Press enter to continue\n"); read(stdin,1);

#
# Second test: fitting a quadratic function
#
quad(x) = (2x+1)^2
trainx,trainy=gendata(quad,0:12)
data=[(trainx,trainy)]

model = Dense(1, 1,x->x^2)
qcost(x, y) = Flux.Losses.mse(model(x), y)
opt = Descent(1e-4/length(trainx))
parameters = params(model)
print("Initial cost: $(qcost(trainx,trainy)), parameters $parameters\n")
for i=1:1000
    mytrain!(qcost, parameters, data, opt)
end
print("Final cost: $(qcost(trainx,trainy)), parameters $parameters\n")

fig=Figure()
ax=fig[1,1]=Axis(fig,title="Simple quadratic fit, y=(2x+1)^2 ($(sum(length.(parameters))) parameters)")
scatter!(ax,vec(trainx),vec(trainy))
lines!(ax,vec(trainx),vec(model(trainx)))
display(fig)
print("Press enter to continue\n"); read(stdin,1);

#
# Third test: Fit a gaussian with a sum of sigmoids
#
gss(x)=4*exp(-(x-1)^2/6)
trainx=collect(-8:.2:8)'
trainy=gss.(trainx)
data=[(trainx,trainy)]
n=40
model=Chain(Dense(1,n,NNlib.sigmoid,init=Flux.glorot_normal),Dense(n,1,bias=false))
qcost(x, y) = Flux.Losses.mse(model(x), y)
opt = AMSGrad()
parameters = params(model)
print("Initial cost: $(qcost(trainx,trainy))\n")
epochs=10000
for i=1:epochs
    mytrain!(qcost, parameters, data, opt)
    print("----- cost: $(qcost(trainx,trainy))\n")
end
print("Final cost after $epochs epochs: $(qcost(trainx,trainy))\n")
fig=Figure();
ax=fig[1,1]=Axis(fig,title="Gaussian fitted with sum of $n sigmoids ($(sum(length.(parameters))) parameters)")
scatter!(ax,vec(trainx),vec(trainy))
lines!(ax,vec(trainx),vec(model(trainx)))
display(fig)
print("Press enter to continue\n"); read(stdin,1);

#
# Fourth test: a complicated function fitted with sigmoids
#
cdr(x) = 4*exp(-(x-5)^2) - exp(-(x+2)^2) + exp(-(x/5)^2)*sin(x)
a=15
trainx=collect(-a:.1:a)'
trainy=cdr.(trainx)
data=[(trainx,trainy)]
n=300
model=Chain(Dense(1,n,NNlib.sigmoid,init=Flux.kaiming_normal),Dense(n,1,NNlib.sigmoid,init=Flux.kaiming_normal))
qcost(x, y) = Flux.Losses.mse(model(x), y)
opt = AMSGrad()
parameters = params(model)
print("Initial cost: $(qcost(trainx,trainy))\n")
epochs=10000
for i=1:epochs
    mytrain!(qcost, parameters, data, opt)
    print("----- cost: $(qcost(trainx,trainy))\n")
end
print("Final cost after $epochs epochs: $(qcost(trainx,trainy))\n")

fig=Figure();
ax=fig[1,1]=Axis(fig,title="More complex fit with $n sigmoid neurons ($(sum(length.(parameters))) parameters)")
scatter!(ax,vec(trainx),vec(trainy))
lines!(ax,vec(trainx),vec(model(trainx)))
display(fig)
print("Press enter to continue\n"); read(stdin,1);


#
# Fifth test: the same function but with a deeper net
#
cdr(x) = 4*exp(-(x-5)^2) - exp(-(x+2)^2) + exp(-(x/5)^2)*sin(x)
a=15
trainx=collect(-a:.1:a)'
trainy=cdr.(trainx)
data=[(trainx,trainy)]
n=10
model=Chain(Dense(1,n,NNlib.sigmoid),Dense(n,n,NNlib.sigmoid),Dense(n,n,NNlib.sigmoid),Dense(n,1))
qcost(x, y) = Flux.Losses.mse(model(x), y)
opt = AMSGrad()
parameters = params(model)
print("Initial cost: $(qcost(trainx,trainy))\n")
epochs=10000
for i=1:epochs
    mytrain!(qcost, parameters, data, opt)
    print("----- cost: $(qcost(trainx,trainy))\n")
end
print("Final cost after $epochs epochs: $(qcost(trainx,trainy))\n")

fig=Figure();
ax=fig[1,1]=Axis(fig,title="More complex fit with 3 hidden layers of $n sigmoid neurons ($(sum(length.(parameters))) parameters)")
scatter!(ax,vec(trainx),vec(trainy))
lines!(ax,vec(trainx),vec(model(trainx)))
display(fig)
print("Press enter to continue\n"); read(stdin,1);
