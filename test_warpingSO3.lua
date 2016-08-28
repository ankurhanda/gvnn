require 'image'
require 'nn'
require 'torch'

dofile('imagewarping.lua')

x = image.loadPNG('linen1.png')
input = torch.Tensor(1,1,240,320)
input[1] = x

r = torch.Tensor(1,3):zero()
r[1][1] = 0.2

t = {input, r}

out_i = tranet:forward(t)

print(#out_i)

out_r = rotation_net:forward(t)

print(#out_r)

out_w = warping_net:forward(t)

print(#out_w)

w = out_w[1]

image.display(x)
image.display(w)

image.save('warped.png', w)
