Experiments on MNIST Data:

Target: Achieving 99.4% accuracy with less than 8k parameters in 15 of less epochs.

I started the process with Code1- where I was only writing the basic framework needed to get mnist data, set up the backpropagation and training modules. 
After this in 2nd Code I observed that the MNIST images are small and there are no patterns or parts of object, just edges and gradient. From this I concluded that there should
be only 1 transition block. I reduced the model parameters slightly- to 21k. There is overfitting and we need to reduce the size.
In the 3rd code, I reduced the parameters to 7.7k keeping it very light.  The model is not learning post 7th epoch. Even the training accuracy is poor. There is mild overfitting. We need to introduce image augmentation and introduce GAP. By GAP, we reduce the parameters needed in last layer, which we can then use to increase in previous layers.
To improve the training accuracy, we can try different optimizer, learning rate and reduce the batch size to avoid the model from getting stuck in local minima.
In 4th code, I introduced the GAP and then tweaked the model slightly. No overfitting now, but the model performance is very poor.
Then I added Batch Normalization and LR on schedule in the 5th code , this led to increase in model performance but still not able to reach 99.4.
In the final code- 6, I introduced One Cycle Learning policy. I experimented with what should be the max_lr. and for what percentage of steps should I increase the LR from, what should be initial LR.
All this experimentation is not shown here. But I found max_lr= 0.5 and pct_start= 0.4 works best. Using these the model was able to reach 99.4 in the last 2 epochs.
