# Lab 2
### Prepare
Create a new notebook

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
```
This model has only 1 parameter `w`. The 
