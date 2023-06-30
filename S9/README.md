Here, we are setting up a modular code to achieve 85% accuracy on CIFAR10 dataset, with less than 200,000 paramaters and RF>=44.
Total parameters= 159,772
To achieve RF =44, I used 16 convolutions.

Implemented 3 transformations from albumentation library- 
HorizontalFlip,ShiftScaleRotate, CoarseDropout.

No Use of Strided or MaxPool.
Used Depthwise and Dilated convolutions in each block.

Kept Dropout= 0.07
Batch_size= 32.
