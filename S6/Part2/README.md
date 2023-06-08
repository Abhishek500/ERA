Ways to achieve 99.4% validation accuracy on MNIST dataset using less than 20k parameters.

To keep parameters under 20k, we need to start with very low number of channels initially.
I am using pyramidal structure- keeping 2-3 layer having same number of channel
For transition, I have used 1x1 and Maxpooling
After each Conv, I am using BatchNorm then ReLU then Dropout,dropout is used only in couple of layer though.
By applying ReLU before Dropout, the Dropout layer can work on the activations produced by ReLU. This ensures that the Dropout process is applied to the activated and meaningful features rather than directly to the raw inputs or unactivated values. Applying Dropout before ReLU may hinder the ReLU activation from fully utilizing the available information and could potentially reduce the model's representational capacity.

I am starting at 0.01 LR, and using scheduler, reducing it by 0.4x every 5 epochs. This will ensure that as reach closer to optimal value, we take smaller steps.


Regarding data transformations:
1. `transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), scale=(0.9, 1.1))`: This transformation randomly applies affine transformations to the input image. It can rotate the image by up to ±20 degrees, translate it horizontally and vertically by up to 10% of the image size, and scale it between 90% and 110% of its original size. These random transformations help augment the training data and introduce variations to improve the model's generalization capability.

2. `transforms.ColorJitter(brightness=0.2, contrast=0.2)`: This transformation applies random color jittering to the input image. It can randomly adjust the brightness and contrast of the image. By introducing variations in brightness and contrast, this transformation helps make the model more robust to changes in lighting conditions during training.

3. `transforms.ToTensor()`: This transformation converts the image from a PIL Image object to a PyTorch Tensor. It also converts the pixel values from the range [0, 255] to the range [0, 1], as Tensors in PyTorch typically have values between 0 and 1.

4. `transforms.Normalize((0.1307,), (0.3081,))`: This transformation normalizes the tensor by subtracting the mean and dividing by the standard deviation. The given values (0.1307 and 0.3081) are the mean and standard deviation of the MNIST dataset, respectively. Normalizing the data helps in making the training process more stable and efficient by reducing the scale of the input features.

By composing these transformations using `transforms.Compose`,I have used them sequentially to the input data,  creating a consistent and standardized input for neural network model.
