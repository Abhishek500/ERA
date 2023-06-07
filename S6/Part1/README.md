Here, key steps under backpropagation is explained. To keep things simple we are using Fully Connected network with 2 input, 2 output and 1 hidden layer with 2 nodes. The initial weights along with the network architecture are shown here 
![image](https://github.com/Abhishek500/ERA/assets/21952545/4af7e8d1-fedb-4f30-b886-ec7233d62464)

Major Steps involved here are:
* Forward Pass: Weights get multiplied with pixel values and then summed up. These are then fed into activation function (here sigmoid). This process is repeated in the output layer also.
* Error Calculation: Feeding input data forward through the network to generate predictions, and then comparing these predictions to the true labels to calculate the loss.
* Backpropagation: It involves computing the gradient of the model's loss function with respect to the weights and the network. The gradients are calculated layer by layer, using the chain rule to propagate the error back through the network. Finally, the weights and biases are adjusted using an optimization algorithm like stochastic gradient descent, which repeats this process iteratively until the model converges to a desired level of performance.

Sharing the screenshot of the excel:
![image](https://github.com/Abhishek500/ERA/assets/21952545/f5f6b73e-2ddf-44a1-a53c-f60dbf8e0caf)
The Loss Graph here is based on learning rate = 1

Now, lets see how the error graph looks like at 0.1 LR.

![image](https://github.com/Abhishek500/ERA/assets/21952545/69b99a71-2194-4ead-b703-68512887b2b4)
We see that the loss is reducing very slowly compared to LR=1

Now, LR= 0.2

![image](https://github.com/Abhishek500/ERA/assets/21952545/d7134926-cda7-4e68-8f24-e7844d0bed21)

Now LR= 0.5

![image](https://github.com/Abhishek500/ERA/assets/21952545/5d225867-b87a-4bc9-91c6-915524fef5e0)

Now LR= 0.8

![image](https://github.com/Abhishek500/ERA/assets/21952545/cfd39f44-cb8c-413c-ae6b-762afaf0d611)

Now LR= 1

![image](https://github.com/Abhishek500/ERA/assets/21952545/29699f91-7b61-4625-a9c4-69cbf622197f)

Now, LR= 2

![image](https://github.com/Abhishek500/ERA/assets/21952545/7e8e8008-b91c-446d-9993-3d4593fc9645)

Here we see that the loss reduces very rapidly

So, as we increase the LR , the loss reaches to optimum value faster.
