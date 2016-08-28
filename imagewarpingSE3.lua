require 'nn'
require 'gvnn'

--dofile('ReverseXYOrder.lua')

concat = nn.ConcatTable()
concat_Rt_depth = nn.ConcatTable()


height = 480--240
width  = 640--320
u0     = 320--160
v0     = 240--120

fx =  480 --240
fy = -480 --240

-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
tranet=nn.Sequential()
tranet:add(nn.SelectTable(1))
tranet:add(nn.Identity())
tranet:add(nn.Transpose({2,3},{3,4}))

-- converts the 6-vector (3-vector so3 for rotation and 3-vector for translation)
Rt_net = nn.Sequential()
Rt_net:add(nn.SelectTable(2))
Rt_net:add(nn.TransformationMatrix3x4SO3(true,false,true))

depth = nn.Sequential()
depth:add(nn.SelectTable(3))

concat_Rt_depth:add(Rt_net)
concat_Rt_depth:add(depth)

Transformation3x4net = nn.Sequential()
Transformation3x4net:add(concat_Rt_depth)
Transformation3x4net:add(nn.Transform3DPoints_Rt(height, width, fx, fy, u0, v0))
Transformation3x4net:add(nn.PinHoleCameraProjectionBHWD(height, width, fx, fy, u0, v0))
Transformation3x4net:add(nn.ReverseXYOrder())

concat:add(tranet)
concat:add(Transformation3x4net)

warping_net = nn.Sequential()
warping_net:add(concat)
warping_net:add(nn.BilinearSamplerBHWD())
warping_net:add(nn.Transpose({3,4},{2,3}))
