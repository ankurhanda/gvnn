require 'gvnn'
require 'torch'
require 'image'

dofile('imagewarpingSE3.lua')

--local height=480
--local width =360

ref_rgb_image   = image.load('iclnuim/rgb/100.png')

ref_depth_image = image.load('iclnuim/depth/100.png')
ref_depth_image = (ref_depth_image*65535)/5000.0

print(ref_rgb_image:size())
print(ref_depth_image:size())

--image.display(ref_rgb_image)
--image.display(ref_depth_image)

data_ref_rgb      = torch.Tensor(1,3,480,640)
data_ref_rgb[1]   = ref_rgb_image

data_ref_depth    = torch.Tensor(1,1,480,640)
data_ref_depth[1] = ref_depth_image

so3_t_vector      = torch.Tensor(1,6):uniform()

-- tx, ty, tz, rx, ry, rz
-- -0.00119339 -0.00449791 -0.00122229 0.00104319 -0.00694122 -0.00333668

--- so3 and translation vector

so3_t_vector[1][1] = 0--  0.00104319
so3_t_vector[1][2] = 0-- -0.00694122
so3_t_vector[1][3] = 0-- -0.00333668

so3_t_vector[1][4] = 0-- -0.00119339
so3_t_vector[1][5] = 0-- -0.00449791
so3_t_vector[1][6] = 0-- -0.00122229

inputTable = {data_ref_rgb:cuda(), so3_t_vector:cuda(), data_ref_depth:cuda()}

outImage = warping_net:cuda():forward(inputTable)

image.display(outImage[1])

--print(torch.max(ref_image))
