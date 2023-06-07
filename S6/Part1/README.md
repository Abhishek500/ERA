Here, key steps under backpropagation is explained. To keep things simple we are using Fully Connected network with 2 input, 2 output and 1 hidden layer with 2 nodes. The initial weights along with the network architecture are shown here 
![image](https://github.com/Abhishek500/ERA/assets/21952545/4af7e8d1-fedb-4f30-b886-ec7233d62464)

Major Steps involved here are:
* Forward Pass: Weights get multiplied with pixel values and then summed up. These are then fed into activation function (here sigmoid). This process is repeated in the output layer also.
* Error Calculation: Feeding input data forward through the network to generate predictions, and then comparing these predictions to the true labels to calculate the loss.
* Backpropagation: It involves computing the gradient of the model's loss function with respect to the weights and the network. Chain rule of derivation comes handy here. This gradient is then used to update the parameters of the network in the opposite direction of the gradient, allowing the model to learn and improve. The process starts with feeding input data forward through the network to generate predictions, and then comparing these predictions to the true labels to calculate the loss. The gradients are then calculated layer by layer, using the chain rule to propagate the error back through the network. Finally, the weights and biases are adjusted using an optimization algorithm like stochastic gradient descent, which repeats this process iteratively until the model converges to a desired level of performance.

Sharing the screenshot of the excel:
![image](https://github.com/Abhishek500/ERA/assets/21952545/f5f6b73e-2ddf-44a1-a53c-f60dbf8e0caf)
The Loss Graph here is based on learning rate = 1

Now, lets see how the error graph looks like at 0.1 LR.
![image](https://github.com/Abhishek500/ERA/assets/21952545/89d733c4-9b9a-4ad3-af6d-59b4f6267bae)
We see that the loss is reducing very slowly compared to LR=1

Now, LR= 0.2
![image](https://github.com/Abhishek500/ERA/assets/21952545/414c2c59-649a-4252-ba0a-fa784257a676)

Now LR= 0.5
![image](https://github.com/Abhishek500/ERA/assets/21952545/4117e44a-c5f0-4978-b55a-d9223335224d)

Now LR= 0.8
![image](https://github.com/Abhishek500/ERA/assets/21952545/95e63a60-8490-4ce8-a2c3-62b06e59bae9)

Now LR= 1
![image](https://github.com/Abhishek500/ERA/assets/21952545/c493a148-b835-4740-a9aa-20692fdfc0da)

Now, LR= 2
![image](https://github.com/Abhishek500/ERA/assets/21952545/ec454779-4a78-4929-985c-d7adcd975042)
Here we see that the loss reduces very rapidly
