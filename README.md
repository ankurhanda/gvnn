Source code for neural network layers performing geometric transformations soon to be added.

Link to the paper [gvnn](http://arxiv.org/pdf/1607.07405.pdf)

gvnn is inspired by the Spatial Transformer Networks (STN) paper that appeared in NIPS in 2015. However, ST were mainly limited to applying only 2D transformations to the input. We added a new set of transformations often needed for manipulating data in 3D geometric computer vision. These include the 3D counterparts of what were used in original STN together with a lot more new transformations.

* SO3 layer - Rotations are expressed in so3 vector (v1, v2, v3)
* SE3 and Sim3 layer 
* Camera Pin-hole projection layer
* 3D Grid Generator
* Per-pixel 2D transformations
    * 2D optical flow
    * 6D Overparamterised optical flow

* Per-pixel 3D transformations
    * 6D SE3/Sim3 transformations
    * 10D transformation

* M-estimators

We plan to make this a comprehensive and complete library to bridge the gap between geometry and deeplearning.


**Can deep learning replace geometry?**

We are still performing large scale experiments on data collected both from real world and our previous work, [SceneNet](http://robotvault.bitbucket.org) to test our various different geometric computer vision algorithms e.g. visual odometry, 3D reconstruction and place recognition. However, this is aimed towards aiding geometry wherever it fails. We certainly believe that convnets can provide stable features for images which in turn can allow matching two images of same scene taken at different times of the day - something not possible with pure geometric methods that either rely on pixel values or SIFT-like features. 
