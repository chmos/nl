# Lab 6 Transformer
## Word embedding
[A tutorial with Pytorch](https://pytorch.org/tutorials/beginner/nlp/word_embeddings_tutorial.html)

## Transfomer
A small sample
```python
tf = nn.Transformer(d_model = 4, nhead = 4)
src = torch.rand((10, 5, 4))
tgt = torch.rand((1, 5, 4))
out = tf(src, tgt)
print(out.shape)
print(out)

for i in tf.children():
    print(str(i))
```

Output:

< torch.Size([1, 5, 4])
< tensor([[[-0.8946, -0.1985, -0.5854,  1.6785],
< [-0.3932, -0.8922, -0.4117,  1.6970],
< [ 0.5631, -0.6444, -1.2369,  1.3182],
< [ 0.0195, -0.8664, -0.7787,  1.6256],
< [ 0.1156, -1.6598,  0.9127,  0.6315]]],
< grad_fn=<NativeLayerNormBackward0>)
