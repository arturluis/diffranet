# torch
import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import torch.utils.data as torch_data

# torchvision
import torchvision.datasets as torchvision_data
import torchvision.transforms as torchvision_transforms
import torch.optim as optimizers

# misc
import math
import os
import pandas as pd
import argparse
import time
import random
import numpy as np

# sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

# config space
import ConfigSpace as CS
import ConfigSpace.hyperparameters as CSH

# bohb
from hpbandster.core.worker import Worker

#logging
import logging
logging.basicConfig(level=logging.INFO)

torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

HEIGHT = 512
WIDTH = 512
TOPOLOGIES = {
    'num_filters'    : [64, 64, 64, 64, 64, 64, 64, 64, 8, 8, 64, 16, 16],
    '1st_conv_stride': [2, 2, 2, 4, 2, 2, 2, 2, 2, 2, 2, 2, 2],
    '2nd_pool_size'  : [9, 13, 7, 7, 7, 15, 8, 7, 7, 7, 9, 9, 7],
    '2nd_pool_stride': [2, 2, 1, 1, 2, 2, 1, 1, 1, 2, 2, 2, 2],
    'block_strides'  : [[1, 2, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2], [1, 2, 2, 2], [2, 2, 2, 2], [1, 2, 3, 3], [1, 2, 2, 2], [1, 2, 2, 2], [1, 2, 2, 3], [1, 2, 2, 2], [1, 2, 2, 2]],
    'final_size'     : [8192, 2048, 2048, 2048, 512, 512, 512, 2048, 6400, 1600, 2048, 2048, 3200]
    }


class PyTorchWorker(Worker):
  # initialize worker
  def __init__(self, gpu_id=0, train_path=None, val_path=None, name='models', **kwargs):
    super().__init__(**kwargs)

    self.gpu_id = gpu_id


    self.train_path = train_path
    self.val_path = val_path
    self.bin_mode = len(os.listdir(val_path)) == 2

    self.name = name

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Worker started")
    print("Using", self.device)



  # Model definition
  def compute(self, config, budget, working_directory, *args, **kwargs):

    # Unpack hyperparameters
    print("Config:\n", config)
    batch_size = config['batch_size']
    lr = config['lr']
    momentum = config['momentum']
    topology = config['topology']
    weight_decay = config['weight_decay']

    # Data preprocessing
    img_mean = self.get_dataset_mean(self.train_path)

    transforms = torchvision_transforms.Compose([
                          torchvision_transforms.Grayscale(),
                          torchvision_transforms.ToTensor(),
                          torchvision_transforms.Normalize(mean=[img_mean], std=[1])
                          ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Training dataset loader
    train_data = torchvision_data.ImageFolder(root=self.train_path, transform=transforms)
    train_loader = torch_data.DataLoader(
                    train_data,
                    batch_size=batch_size,
                    shuffle=True,
                    **kwargs
                    )
    train_size = len(train_data.imgs)

    # Validation dataset loader
    val_data = torchvision_data.ImageFolder(root=self.val_path, transform=transforms)
    val_loader = torch_data.DataLoader(
                    val_data,
                    batch_size=batch_size,
                    shuffle=False,
                    **kwargs
                    )
    val_size = len(val_data.imgs)


    # create model
    model = resnet50(num_classes=5, topology=topology)

    device = self.device
    if (torch.cuda.device_count() > 1):
      model = nn.DataParallel(model)
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optimizers.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay
            )

    # train network
    for epoch in range(int(budget)):
      train_loss = 0.0
      correct = 0
      count = 0
      self.adjust_learning_rate(optimizer, epoch, decay_epochs=10)
      print("worker:", self.gpu_id, "starting epoch", epoch)

      for batch_ix, data in enumerate(train_loader):
        count += 1

        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, 1) # torch.max() returns a tuple (max, argmax)
        correct += (predicted == labels).sum().item()
        train_loss += loss.item()

    train_accuracy = 100*(correct/train_size)
    train_loss = train_loss/train_size
    validation_accuracy, validation_loss = self.validate(val_loader, model, loss_function)

    os.makedirs(self.name, exist_ok=True)
    model_filename = self.name + '/worker_' + str(self.gpu_id) + '_tacc_' + str(train_accuracy) + '_vacc_' + str(validation_accuracy) + '_tstamp_' + str(time.time()) + '_model.tar'
    self.save_model(model_filename, budget, model, optimizer)

    return ({
      'loss': 1-validation_accuracy,
      'info': { 'train accuracy': train_accuracy,
            'train_loss': train_loss,
            'validation accuracy': validation_accuracy,
            'validation_loss': validation_loss,
            'learning rate': optimizer.param_groups[0]['lr'],
            'topology': topology,
            'budget': int(budget)
          }

    })

  def validate(self, loader, model, loss_function):
    correct = 0
    total = 0
    loss = 0
    device = self.device
    model.eval()
    y_pred = []
    y_true = []
    with torch.no_grad():
      for data in loader:
        images, labels = data
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        if self.bin_mode:
          for pred, label in zip(predicted, labels):
            pred_value = 0 if (pred.item() <= 1) else 1
            label_value = label.item()
            correct += (pred_value == label_value)
        else:
          correct += (predicted == labels).sum().item()
        loss += loss_function(outputs, labels).item()
    acc = correct / len(loader.dataset.imgs)
    loss /= len(loader.dataset.imgs)
    model.train()
    return acc, loss #}}}

  def get_dataset_mean(self, train_path):
    transforms = torchvision_transforms.Compose([
                                               torchvision_transforms.Grayscale(),
                                               torchvision_transforms.ToTensor()
                                               ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    train_data = torchvision_data.ImageFolder(root=train_path, transform=transforms)
    train_loader = torch_data.DataLoader(
                                        train_data,
                                        batch_size=1,
                                        shuffle=True,
                                        **kwargs
                                        )
    train_size = len(train_data.imgs)

    img_mean = np.zeros((HEIGHT,WIDTH))
    for data in train_loader:
        image, label = data
        img_mean += image.view(HEIGHT,WIDTH)

    img_mean = img_mean/train_size
    final_mean = np.mean(img_mean)
    return final_mean


  def adjust_learning_rate(self, optimizer, epoch, decay_epochs=10, decay_rate=10):

      lr = optimizer.param_groups[0]['lr']
      if (epoch % decay_epochs == 0):
          lr = lr/decay_rate

      for param_group in optimizer.param_groups:
          param_group['lr'] = lr


  def save_model(self, filename, epoch, model, optimizer):
      state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
        }
      torch.save(state, filename)


  @staticmethod
  def get_configspace():
    cs = CS.ConfigurationSpace()

    topology = CSH.CategoricalHyperparameter('topology', [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
    lr = CSH.UniformFloatHyperparameter('lr', lower=1e-4, upper=1, default_value='1e-1', log=True)
    momentum = CSH.UniformFloatHyperparameter('momentum', lower=0.5, upper=1, default_value=0.9, log=False)
    weight_decay = CSH.UniformFloatHyperparameter('weight_decay', lower=0.00001, upper=0.00005, default_value=0.00001, log=False)
    batch_size = CSH.CategoricalHyperparameter('batch_size', [1, 2, 4, 8, 16], default_value=8)

    cs.add_hyperparameters([
        topology,
        lr,
        momentum,
        weight_decay,
        batch_size
        ])


    return cs


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes, topology=3):

        # Network topology parameters
        filters = TOPOLOGIES['num_filters'][topology]
        conv1_kernel = 7
        conv1_stride = TOPOLOGIES['1st_conv_stride'][topology]
        pool1_kernel = 3
        pool1_stride = 2
        pool2_kernel = TOPOLOGIES['2nd_pool_size'][topology]
        pool2_stride = TOPOLOGIES['2nd_pool_stride'][topology]
        block_sizes = layers
        block_strides = TOPOLOGIES['block_strides'][topology]
        block_filters = [filters, filters*2, filters*4, filters*8]
        final_size = TOPOLOGIES['final_size'][topology]

        print(
            "Topology:      %d"
            "\nnum_filters:   %d"
            "\nconv1_kernel:  %d"
            "\nconv1_stride:  %d"
            "\npool1_kernel:  %d"
            "\npool1_stride:  %d"
            "\npool2_kernel:  %d"
            "\npool2_stride:  %d"
            "\nblock_sizes:   %s"
            "\nblock_strides: %s"
            "\nblock_filters: %s"
            "\nfinal_size:    %d"
            %(
              topology+1,
              filters,
              conv1_kernel,
              conv1_stride,
              pool1_kernel,
              pool1_stride,
              pool2_kernel,
              pool2_stride,
              tuple(block_sizes),
              tuple(block_strides),
              tuple(block_filters),
              final_size
              )
            )


        self.inplanes = filters
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, filters, kernel_size=conv1_kernel, stride=conv1_stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=pool1_kernel, stride=pool1_stride, padding=1)
        self.layer1 = self._make_layer(block, block_filters[0], block_sizes[0], stride=block_strides[0])
        self.layer2 = self._make_layer(block, block_filters[1], block_sizes[1], stride=block_strides[1])
        self.layer3 = self._make_layer(block, block_filters[2], block_sizes[2], stride=block_strides[2])
        self.layer4 = self._make_layer(block, block_filters[3], block_sizes[3], stride=block_strides[3])
        self.avgpool = nn.AvgPool2d(pool2_kernel, stride=pool2_stride)
        self.fc = nn.Linear(final_size * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(topology=3, **kwargs):

    model = ResNet(Bottleneck, [3, 4, 6, 3], topology=topology, **kwargs)
    return model


# short code to test worker
if __name__ == "__main__":
  worker = PyTorchWorker(run_id='0')
  cs = worker.get_configspace()

  config = cs.sample_configuration().get_dictionary()
  print(config)
  res = worker.compute(config=config, budget=1, working_directory='.')
  print(res)
