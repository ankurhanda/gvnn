Link to the paper [gvnn](http://arxiv.org/pdf/1607.07405.pdf)

gvnn is inspired by the Spatial Transformer Networks (STN) paper that appeared in NIPS in 2015. However, ST were mainly limited to applying only 2D transformations to the input. We added a new set of transformations often needed for manipulating data in 3D geometric computer vision. These include the 3D counterparts of what were used in original STN together with a lot more new transformations.

* SO3 layer   - Rotations are expressed in so3 vector (v1, v2, v3)
* Euler layer - Rotations are also expressed in euler angles
* SE3 and Sim3 layer 
* Camera Pin-hole projection layer
* 3D Grid Generator
* Per-pixel 2D transformations
    * 2D optical flow
    * 6D Overparameterised optical flow
    * Per-pixel SE(2)
    * Slanted plane disparity

* Per-pixel 3D transformations
    * 6D SE3/Sim3 transformations
    * 10D transformation

* M-estimators

We plan to make this a comprehensive and complete library to bridge the gap between geometry and deeplearning.


We are still performing large scale experiments on data collected both from real world and our previous work, [SceneNet](http://robotvault.bitbucket.org) to test our various different geometric computer vision algorithms e.g. dense image registration, 3D reconstruction and place recognition.


#Installation 

luarocks make gvnn-scm-1.rockspec

#SO3 Layer 
To set up 3D rotation warping, you first need to homogenise the x,y positions to [x, y, 1]^T, apply the inverse camera calibration matrix to get the ray in 3D. This ray is rotated with the rotation and then backprojected into the 2D plane with PinHoleCameraProjection layer and interpolated with bilinear interpolation.

```
require 'nn'
require 'gvnn'

concat = nn.ConcatTable()

height = 240
width  = 320
u0     = 160
v0     = 120

fx = 240
fy = 240

-- first branch is there to transpose inputs to BHWD, for the bilinear sampler
tranet=nn.Sequential()
tranet:add(nn.SelectTable(1))
tranet:add(nn.Identity())
tranet:add(nn.Transpose({2,3},{3,4}))

rotation_net = nn.Sequential()
rotation_net:add(nn.SelectTable(2))
rotation_net:add(nn.TransformationRotationSO3())
rotation_net:add(nn.Transform3DPoints_R(height, width, fx, fy, u0, v0))
rotation_net:add(nn.PinHoleCameraProjectionBHWD(height, width, fx, fy, u0, v0))
rotation_net:add(nn.ReverseXYOrder())

concat:add(tranet)
concat:add(rotation_net)

warping_net = nn.Sequential()
warping_net:add(concat)
warping_net:add(nn.BilinearSamplerBHWD())
warping_net:add(nn.Transpose({3,4},{2,3}))

```

This is how to use the previous network to warp and plot the image

```
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
```



![Montage-0](assets/so3_rot_example.png)

#License 
GPL

If you use the code, please consider citing the following 
```
@inproceedings{PatrauceanHC16,
  author    = {Ankur Handa and 
               Michael Bloesch and 
               Viorica P{\u a}tr{\u a}ucean and
               Simon Stent and
               John McCormac and
               Andrew Davison},
  title     = {gvnn: Neural Network Library for Geometric Computer Vision},
  booktitle = {arXiv:1607.07405},
  year      = {2016}
}
```
```
@Misc{STNImplementation,
    author = {Moodstocks},
    title={{Open Source Implementation of Spatial Transformer Networks}},
    howpublished={URL https://github.com/qassemoquab/stnbhwd},
    year={2015}
}
```
