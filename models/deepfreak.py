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
import numpy as np

# sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


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

# check if there is gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# PyTorch ResNet
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

    def __init__(self, block, layers, num_classes, topology=7):

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
            "Topology:        %d"
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


def resnet50(topology=7, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=5, topology=topology)
    return model


def validate(loader, model, loss_function, bin_mode=False):
    model.eval()
    correct = 0
    total = 0
    loss = 0
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            if bin_mode:
              for pred, label in zip(predicted, labels):
                pred_value = 0 if (pred.item() <= 1) else 1
                label_value = label.item()
                correct += (pred_value == label_value)
            else:
              correct += (predicted == labels).sum().item()
            loss += loss_function(outputs, labels).item()
    acc = 100 * correct / len(loader.dataset.imgs)
    loss /= len(loader.dataset.imgs)
    model.train()
    return acc, loss


def save_model(filename, epoch, model, optimizer):
    state = {
    'epoch': epoch,
    'state_dict': model.state_dict(),
    'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, filename)


def adjust_learning_rate(optimizer, epoch, decay_epochs=10, decay_rate=10):

  lr = optimizer.param_groups[0]['lr']
  if (epoch % decay_epochs == 0):
    lr = lr/decay_rate

  for param_group in optimizer.param_groups:
    param_group['lr'] = lr


def confusion_mat(dataset, model, batch_size, test=0):
    model.eval()
    # create ground truth
    y_true = []
    for sample in dataset.samples:
      y_true.append(sample[1])


    loader = torch_data.DataLoader(
                                dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                )

    # get predictions
    y_pred = []
    with torch.no_grad():
        for data in loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            y_pred += predicted.tolist()
    confusion_mat = confusion_matrix(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average=None)
    recall = recall_score(y_true, y_pred, average=None)
    print(confusion_mat)
    print("Precision:", precision)
    print("Recall:", recall)
    model.train()


def get_dataset_mean(train_path):
    transforms = torchvision_transforms.Compose([
                                                torchvision_transforms.Grayscale(),
                                                torchvision_transforms.ToTensor()
                                                ])
    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Training dataset loader
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
    final_std = np.std(img_mean)
    return final_mean, final_std


def main(args):
    train_path = args.train_path
    val_path = args.val_path
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    checkpoint_epochs = args.checkpoint_epochs
    mat_epochs = args.mat_epochs
    log_epochs = args.log_epochs
    output_folder = args.output_folder
    start_epoch = args.start_epoch
    lrs_rates = args.lrs_rates
    lrs_epochs = args.lrs_epochs
    weight_decay = args.weight_decay
    lr = args.learning_rate
    momentum = args.momentum
    topology = args.topology

    if (len(os.listdir(val_path)) == 2):
      bin_mode = True
    else:
      bin_mode = False

    print(
        "Training set:", train_path,
        "\nValidation set:", val_path,
        "\nBatch_size:", batch_size,
        "\nNumber of epochs:", num_epochs,
        "\nEpochs per checkpoint:", checkpoint_epochs,
        "\nEpochs per confusion matrix:", mat_epochs,
        "\nEpochs per results log:", log_epochs,
        "\nOutput folder:", output_folder,
        "\nLearning rates:", lrs_rates,
        "\nDecay epochs:", lrs_epochs,
        "\nWeight decay:", weight_decay,
        "\nLearning rate:", lr,
        "\nMomentum:", momentum,
        "\nTopology:", topology+1
        )

    img_mean, img_std = get_dataset_mean(train_path)

    # Data preprocessing
    train_transforms = torchvision_transforms.Compose([
      torchvision_transforms.Grayscale(),
      torchvision_transforms.ToTensor(),
      torchvision_transforms.Normalize(mean=[img_mean], std=[1])
      ])

    val_transforms = torchvision_transforms.Compose([
      torchvision_transforms.Grayscale(),
      torchvision_transforms.ToTensor(),
      torchvision_transforms.Normalize(mean=[img_mean], std=[1])
      ])

    kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

    # Training dataset loader
    train_data = torchvision_data.ImageFolder(root=train_path, transform=train_transforms)
    train_loader = torch_data.DataLoader(
                                    train_data,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    **kwargs
                                    )
    train_size = len(train_data.imgs)

    # Validation dataset loader
    val_data = torchvision_data.ImageFolder(root=val_path, transform=val_transforms)
    val_loader = torch_data.DataLoader(
                                    val_data,
                                    batch_size=batch_size,
                                    shuffle=False,
                                    **kwargs
                                    )
    val_size = len(val_data.imgs)

    print("Found", train_size, "training images belonging to", len(train_data.classes), "classes")
    print("Found", val_size, "validation images belonging to", len(val_data.classes), "classes")

    model = resnet50(num_classes=5, topology=topology)

    # Send to gpus
    if (torch.cuda.device_count() > 1):
        model = nn.DataParallel(model)
    model.to(device)

    loss_function = nn.CrossEntropyLoss()
    optimizer = optimizers.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_schedule = {
        'epochs': lrs_epochs,
        'rates' : lrs_rates
        }
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)
    checkpoint_file = output_folder + 'model_checkpoint.tar'

    # If a checkpoint exists, load it
    if (os.path.isfile(checkpoint_file)):
        checkpoint = torch.load(checkpoint_file)
        start_epoch = checkpoint['epoch'] + 1 # checkpoint saves the latest epoch done
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("Loaded checkpoint '{}' (epoch {})"
                .format(checkpoint_file, checkpoint['epoch']))

    # Train the network
    results = {
        'train_accs': [],
        'val_accs': [],
        'train_losses': [],
        'val_losses': [],
        'time': [],
        'learning_rate': [],
        'precision': [],
        'recall': [],
        'f1score': [],
        }
    results_file = output_folder + 'results.log'
    best_val_acc = 0
    print("Starting training...")
    for epoch in range(start_epoch, num_epochs+1):
        t0 = time.time()
        y_pred = []
        y_true = []
        train_loss = 0.0
        correct = 0
        adjust_learning_rate(optimizer, epoch, decay_epochs=10)
        for batch_ix, data in enumerate(train_loader):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
            train_loss += loss.item()

            y_pred += predicted.tolist()
            y_true += labels.tolist()

        save_model(checkpoint_file, epoch, model, optimizer)

        train_acc = 100*(correct/train_size)
        val_acc, val_loss = validate(val_loader, model, loss_function, bin_mode)
        train_loss /= train_size
        lr = optimizer.param_groups[0]['lr']
        precision = precision_score(y_true, y_pred, average=None)
        recall = recall_score(y_true, y_pred, average=None)
        f1score = f1_score(y_true, y_pred, average=None)


        # print confusion matrix
        if (epoch % mat_epochs == 0):
            confusion_mat(val_data, model, batch_size)
            confusion_mat(train_data, model, batch_size)
        if (epoch % log_epochs == 0):
            results_df = pd.DataFrame.from_dict(results)
            results_df.to_csv(results_file, float_format='%.4f', index=False)

        # print epoch statistics
        epoch_time = time.time() - t0
        print(
            "epoch: %d, "
            "time: %.2f, "
            "learning rate: %g, "
            "training loss: %.4f, "
            "training accuracy: %.2f%%, "
            "validation loss: %.4f, "
            "validation accuracy: %.2f%%"
            "\nprecision: %s"
            "\nrecall: %s"
            "\nf1-score: %s"
               %(
                epoch,
                epoch_time,
                lr,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                tuple(precision),
                tuple(recall),
                tuple(f1score)
                )
            )

        print("Comparing results:", val_acc, best_val_acc)
        if val_acc > best_val_acc:
          print("Updating best model. Validation accuracy:", best_val_acc, " -> ", val_acc)
          best_model_file = output_folder + 'best_model.tar'
          best_val_acc = val_acc
          save_model(best_model_file, epoch, model, optimizer)


        results['train_accs'].append(train_acc)
        results['train_losses'].append(train_loss)
        results['val_accs'].append(val_acc)
        results['val_losses'].append(val_loss)
        results['time'].append(epoch_time)
        results['learning_rate'].append(lr)
        results['precision'].append(precision)
        results['recall'].append(recall)
        results['f1score'].append(f1score)


if __name__== "__main__":
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    parser = argparse.ArgumentParser(description='Train our ResNet network')
    parser.add_argument('--train_path', dest='train_path', action='store', default="../data/synthetic/train/")
    parser.add_argument('--val_path', dest='val_path', action='store', default="../data/real_preprocessed/validation/")
    parser.add_argument('--batch_size', dest='batch_size', action='store', default=8, type=int)
    parser.add_argument('--start_epoch', dest='start_epoch', action='store', default=1, type=int)
    parser.add_argument('--num_epochs', dest='num_epochs', action='store', default=180, type=int)
    parser.add_argument('--checkpoint_epochs', dest='checkpoint_epochs', action='store', default=5, type=int)
    parser.add_argument('--mat_epochs', dest='mat_epochs', action='store', default=5, type=int)
    parser.add_argument('--log_epochs', dest='log_epochs', action='store', default=5, type=int)
    parser.add_argument('--output', dest='output_folder', action='store', default='deepfreak_output/')
    parser.add_argument('--learning_rate', dest='learning_rate', action='store', default=0.0048311, type=float)
    parser.add_argument('--lrs-rates', dest='lrs_rates', action='store', default=[0.1, 0.01, 0.001, 0.0001], type=float, nargs='+')
    parser.add_argument('--lrs-epochs', dest='lrs_epochs', action='store', default=[90, 120, 180], type=int, nargs='+')
    parser.add_argument('--weight_decay', dest='weight_decay', action='store', default=0.000021462, type=float)
    parser.add_argument('--momentum', dest='momentum', action='store', default=0.88172, type=float)
    parser.add_argument('--topology', dest='topology', action='store', default=9, type=int)



    args = parser.parse_args()
    main(args)

