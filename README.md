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
