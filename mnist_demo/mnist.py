import os
from datasets import load_dataset, load_from_disk, Dataset
from tinygrad import Device, Tensor, nn, TinyJit
import numpy as np
import time

dev = Device.DEFAULT
# print(dev)

def pre_data():
  mnist_dir = "mnist_datasets"
  # download and load mnist datasets from huggingface
  if not os.path.exists(f'./{mnist_dir}'):
    ds = load_dataset("ylecun/mnist")
    # saving the datasets
    ds.save_to_disk(mnist_dir)
  else:
    ds = load_from_disk(mnist_dir)

  train_images = np.array([np.array(img) for img in ds['train']['image']])
  train_labels = np.array(ds['train']['label'])
  test_images = np.array([np.array(img) for img in ds['test']['image']])
  test_labels = np.array(ds['test']['label'])

  train_images = Tensor((train_images.astype(np.float32) / 255.0)[:, None, :, :], device=dev)
  train_labels = Tensor(train_labels.astype(np.int64), device=dev)
  test_images = Tensor((test_images.astype(np.float32) / 255.0)[:, None, :, :], device=dev, requires_grad=False)
  test_labels = Tensor(test_labels.astype(np.int64), device=dev)
  
  return train_images, train_labels, test_images, test_labels

# print(train_images.shape)
# print(train_labels.shape)


class Model:
  def __init__(self):
    self.l1 = nn.Conv2d(1, 32, kernel_size=(3,3))
    self.l2 = nn.Conv2d(32, 64, kernel_size=(3,3))
    self.l3 = nn.Linear(1600, 10)
    
  def __call__(self, x: Tensor) -> Tensor:
    x = self.l1(x).relu().max_pool2d((2, 2))
    x = self.l2(x).relu().max_pool2d((2, 2))
    return self.l3(x.flatten(1).dropout(0.5))
    

def step():
  Tensor.training = True
  samples = Tensor.randint(batch_size, high=X_train.shape[0])
  X, Y = X_train[samples], Y_train[samples]
  optim.zero_grad()
  loss = model(X).sparse_categorical_crossentropy(Y).backward()
  optim.step()
  return loss


jit_step = TinyJit(step)

def save_weights(model, path):
  np.savez(path,
           l1_w = model.l1.weight.numpy(), l1_b=model.l1.bias.numpy(),
           l2_w = model.l2.weight.numpy(), l2_b=model.l2.bias.numpy(),
           l3_w = model.l3.weight.numpy(), l3_b=model.l3.bias.numpy(),
  )

def load_weights(model, path, device):
  ws = np.load(path)
  dev = device

  model.l1.weight.assign(Tensor(ws["l1_w"], device=dev))
  model.l1.bias  .assign(Tensor(ws["l1_b"], device=dev))
  model.l2.weight.assign(Tensor(ws["l2_w"], device=dev))
  model.l2.bias  .assign(Tensor(ws["l2_b"], device=dev))
  model.l3.weight.assign(Tensor(ws["l3_w"], device=dev))
  model.l3.bias  .assign(Tensor(ws["l3_b"], device=dev))

model_name = "mnist_tinygrad.npz"


RESET = "\x1b[0m"

def print_blocks(img):
  for row in img:
    line = []
    for v in row:
      g = max(0, min(255, int(v*255)))
      line.append(f"\x1b[48;2;{g};{g};{g}m  ")
    print("".join(line) + RESET)

def print_all(img_x):
  print_blocks(img_x[0].numpy())
  x = img_x.reshape(1, 1, 28, 28)
  Tensor.training = False
  logits = model(x)
  pred = logits.argmax(axis=1).numpy()[0]
  print(f"pred:{pred}", end=" ")
  
if __name__ == "__main__":
  try:
    model = Model()
    load_weights(model, model_name, dev)
    _, _, X_test, Y_test = pre_data()
    for i in range(2000, 2100, 5):
      X, Y = X_test[i], Y_test[i]
      print_all(X) 
      print(f"raw value:{Y.numpy()}")
      time.sleep(1)

  except FileNotFoundError:
    X_train, Y_train, X_test, Y_test = pre_data()
    model = Model()
    optim = nn.optim.Adam(nn.state.get_parameters(model))
    batch_size = 128 
    for i in range(7000):
      loss = jit_step()
      if (i+1) % 100 == 0:
        Tensor.training = False
        acc = (model(X_test).argmax(axis=1) == Y_test).mean().item()
        print(f"step {i:4d}, loss {loss.item():.2f}, acc {acc*100.:.2f}%")
        Tensor.training = True
    save_weights(model, model_name)
    print("save_weights")
    
  
