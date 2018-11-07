### Primary capsule
Applies `keras.layers.Conv2D` to input. 
Then reshapes output of convolution `(batch, height, width, channels)` to `(batch,num_of_capsules,conv_units)`.
At last applies [squash function](https://github.com/zx-/CapsNet/blob/master/layers/helpers.py#L9) as activation function.

Check [documentation for more info](https://github.com/zx-/CapsNet/blob/master/layers/primary_caps.py#L6)

![Primary Caps](https://github.com/zx-/CapsNet/blob/master/images/primary_caps.PNG)

### Capsules
Takes lower level capsule layers as input. 
First computes [predictions](https://github.com/zx-/CapsNet/blob/master/layers/capsule.py#L74) 
and then applies [routing algorithm](https://github.com/zx-/CapsNet/blob/master/layers/capsule.py#L115).

Check [documentation for more info](https://github.com/zx-/CapsNet/blob/master/layers/capsule.py#L154)

![Capsule](https://github.com/zx-/CapsNet/blob/master/images/capsule.PNG)

### Reconstruction
Simple dense network is used for image reconstrucion.
![Reconstruction](https://github.com/zx-/CapsNet/blob/master/images/reconstruction.PNG)
