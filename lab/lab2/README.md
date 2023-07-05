# Lab 2
### Prepare
Create a new notebook and import the related packages:
```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
```

### Create a customized module
This snippet creates a module (or layer) that outputs the sin of the input

$$
y = \sin(x)
$$

```python
# create a customized module
class Sin(nn.Module):
    def __init__(self):
        super(Sin, self).__init__()
        # -- must register this as parameter
        self.w = torch.nn.Parameter(torch.rand(1))

    # xb is a bacth of x_i, i.e., x1, x2, .., xn
    def forward(self, input):
        output = torch.sin(input * self.w)
        # output.requires_grad=True
        return output

# sn is an instance of the model
sn = Sin()
print(sn)
```
This model has only 1 parameter `w`. The line 

`self.w = torch.nn.Parameter(torch.rand(1))` 

declare it is a parameter, and initialized as a random number.

### Train this model
```python
# generate a sample batch: 20*2 input tensors, 20*2 output tensors
# there are 20 samples in a batch
def gen_samples(k):
    x = np.random.uniform(-10, 10, [20, 2])
    y = np.sin(k * x)
    xb = torch.tensor(x.transpose().astype('float32'))
    yb = torch.tensor(y.transpose().astype('float32'))
    return xb, yb

# train the model once, and return the loss in this epoch
def train_epoch(batch_in, batch_out, optimizer, model):
    input = batch_in
    output = model(input)
    loss = nn.functional.l1_loss(output, batch_out)
    model.zero_grad()
    loss.backward()
    optimizer.step()
    return loss

# instance of the model
sn = Sin()
print(sn)
# optimizer for training
optimizer = torch.optim.SGD(sn.parameters(), lr = 1e-4, 
                            weight_decay = 1e-3, momentum = 0.9)

# ep records losses during training
ep = []

k = 3.1415926 / 4
xb, yb = gen_samples(k)

for i in range(600):
  # xb, yb = gen_samples(k)
  loss = train_epoch(xb, yb, optimizer, sn)
  if i % 10 == 0:
      # -- you can not plot tensors. Need to
      # change them into np vectors
      ep.append(loss.detach().numpy())

# show the trained parameters and plot losses
print(sn.w, 'vs.', k)
plt.figure()
plt.plot(np.arange(len(ep)), ep)
```

### Practice 1
1. Create a model with 2 parameters $k_1$, $k_2$, that outputs $y = \sin(k_1 x_1 + k_2  x_2)$.
   You can treat $k1,k2$ as one tensor with 2 entries

2. Suppose there are $N$ samples, each sample is a 2-entries tensor, as shown in the above section. What dimensions does the output have? Write a function to generate the input and output samples.
3. Train the model with the generated samples, compare the trained $k1, k2$ with the $k1,k2$ you used to generate the samples, and plot the losses
4. In the line defining the optimizer, change the `weight_decay`, e.g., to $10^{-2}$. Repeat the step 3, what conclusion do you get?

### Practise 2
Now create a module with 2 layers. The first layer is a linear layer, and the second layer is the Sin module you have just created. Now the output $z$ is given by

$$
(y_1, y_2) = (x_1, x_2)A^T + b 
$$

$$
z = k_1 y_1 + k_2 y_2
$$

Create the model, training samples, train and plot losses

### Upload your notebook
Submit your notebook to your GitHub. You can use use "File -> Download" to download a notebook to your local machine. And then directly drag it after you click "Add file" in your GitHub page.

### Sample solution
[Practise](lab2.ipynb)


