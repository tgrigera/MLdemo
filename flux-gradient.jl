# flux-gradient.jl --- Trying taking derivatives with Flux (actually, Zygote)
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


#
# Taking gradients #1: one variable
#
f(x) = log(x)
df(x) = gradient(f,x)[1]
fig=Figure()
ax=fig[1,1]=Axis(fig,title="Taking derivatives with flux: log(x)")
x=collect(1:10)
lines!(ax,x,1 ./x,label="Analytic derivative")
scatter!(ax,x,df.(x),label="Derivative by flux")
axislegend(ax,framevisible=false,position=:rt)
display(fig)
print("Press enter to continue\n"); read(stdin,1);

#
# Taking gradients #2: two variables
#
g(x,b)=x^b
∂g(x,b)=gradient(g,x,b)
fig=Figure()
ax=fig[1,1]=Axis(fig,title="Taking derivatives with flux: x^b")
x=collect(1:10)
b=3
lines!(ax,x,b*x.^(b-1),label="∂f/∂x, analyitic")
scatter!(ax,x,first.(∂g.(x,b)),label="∂f/∂x, flux")
b=collect(1:10)
x=2
lines!(ax,b,log(x)*x.^b,label="∂f/∂b, analyitic")
scatter!(ax,b,map(x->x[2],∂g.(x,b)),label="∂f/∂b, flux")
axislegend(ax,framevisible=false,position=:lt)
display(fig)
print("Press enter to continue\n"); read(stdin,1);

#
# Taking gradients #3: x^n defined with a for loop
#
function h(x,n)  y=1; for i=1:n y*=x; end; return y; end
∂h(x,n) = gradient(h,x,n)
# The gradient wrt n is not defined
∂h(2,3)
fig=Figure();
ax=fig[1,1]=Axis(fig,title="Taking derivatives with flux: x^n with for-loop")
n=4
x=collect(-10:10)
lines!(ax,x,x.^n,label="x^$n with power")
scatter!(ax,x,h.(x,n),label="x^$n with for-loop")
lines!(ax,x,n*x.^(n-1),label="derivative, $(n)x^$(n-1), analytic")
scatter!(ax,x,first.(∂h.(x,n)),label="derivative, flux")
axislegend(ax,framevisible=false,position=:ct)
display(fig)

# WARNING, the following does not work, apparently y=x fools gradient()
function h(x,n)  y=x; for i=1:n-1 y*=x; end; return y; end
# This way it does work
function h(x,n)  y=copy(x); for i=1:n-1 y*=x; end; return y; end
