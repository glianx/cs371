from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as fct
from math import prod
from types import SimpleNamespace
import pickle
from tqdm import tqdm
import os

def parameter_count(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)


class SimpleNet(nn.Module):
    def __init__(self, input_shape, n_classes, conv_channels, fc_channels):
        super().__init__()
        kernel_size = 5
        padding = kernel_size // 2
        strides = [2, 2]
        self.conv1 = nn.Conv1d(
            in_channels=input_shape[1], out_channels=conv_channels[0],
            kernel_size=kernel_size, stride=strides[0], padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_channels[0], out_channels=conv_channels[1],
            kernel_size=kernel_size, stride=strides[1], padding=padding)
        m = conv_channels[-1] * (input_shape[-1] // prod(strides))
        self.fc1 = nn.Linear(m, fc_channels[0])
        self.fc2 = nn.Linear(fc_channels[0], fc_channels[1])
        self.fc3 = nn.Linear(fc_channels[1], n_classes)

    def forward(self, x):
        x = fct.relu(self.conv1(x))
        x = fct.relu(self.conv2(x))
        x = torch.flatten(x, 1)
        x = fct.relu(self.fc1(x))
        x = fct.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class MusicDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = self.data[idx]
        label = self.labels[idx]
        return item.view(1, *item.shape), label


def music_loaders(file_name, batch_size=4):
    base = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base, file_name)
    with open(file_path, 'rb') as file:
        data = pickle.load(file)
    train_set = MusicDataset(data['train']['x'], data['train']['y'])
    validate_set = MusicDataset(data['validate']['x'], data['validate']['y'])
    test_set = MusicDataset(data['test']['x'], data['test']['y'])

    classes = data['genres']

    loader = torch.utils.data.DataLoader(
        train_set, batch_size=1, shuffle=False, num_workers=2
    )
    iterator = iter(loader)
    snippet, _ = next(iterator)
    shape = snippet.size()

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    validate_loader = torch.utils.data.DataLoader(
        validate_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return SimpleNamespace(
        train=train_loader, validate=validate_loader, test=test_loader,
        classes=classes, input_shape=shape
    )


def train(
        model, loader, epochs, start_epoch, learning_rate=0.0001,
        momentum=0.9, print_interval=2000
):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, momentum=momentum
    )

    fmt = '[{:2d}, {:5d}] risk over last {} mini-batches: {:.3f}'
    for epoch in range(epochs):
        running_risk = 0.
        desc = 'epoch {}/{}'.format(epoch + 1, epochs)
        for i, batch in tqdm(enumerate(loader, 0), desc=desc, leave=False):
            inputs, labels = batch
            optimizer.zero_grad()

            predictions = model(inputs)
            risk = loss_function(predictions, labels)
            risk.backward()
            optimizer.step()

            running_risk += risk.item()
            if i % print_interval == print_interval - 1:
                block_risk = running_risk / print_interval
                ep = epoch + start_epoch
                print(fmt.format(ep + 1, i + 1, print_interval, block_risk))
                running_risk = 0.


def evaluate(network, loader, set_name):
    correct, total = 0, 0
    with torch.no_grad():
        for batch in tqdm(loader):
            images, labels = batch
            outputs = network(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print('{} accuracy {:.2f} percent'.format(set_name, accuracy))
    return accuracy


def experiment(loaders, conv_channels, fc_channels):
    assert len(conv_channels) == 2 and len(fc_channels) == 2, 'two layers of each type, please'
    model = SimpleNet(
        loaders.input_shape, len(loaders.classes),
        conv_channels=conv_channels, fc_channels=fc_channels)
    print('Model:')
    print(model)
    print('{} weights'.format(parameter_count(model)))

    accuracies = {'epoch': [], 'train': [], 'validate': [], 'weight count': 0}
    epochs_per_group = 5
    print('training')
    for group in range(20):
        print('{}-epoch group number {}'.format(epochs_per_group, group + 1))
        se = group * epochs_per_group
        train(model, loaders.train, epochs=epochs_per_group, start_epoch=se)

        e = se + epochs_per_group
        print('evaluating after epoch {}'.format(e))
        a_train = evaluate(model, loaders.train, 'training')
        a_validate = evaluate(model, loaders.validate, 'validation')

        accuracies['epoch'].append(e)
        accuracies['train'].append(a_train)
        accuracies['validate'].append(a_validate)
        accuracies['weight count'] = parameter_count(model)

    channels_string = '_'.join([str(c) for c in conv] + [str(c) for c in fc])
    file_name = 'accuracies_{}.pkl'.format(channels_string)
    with open(file_name, 'wb') as file:
        pickle.dump(accuracies, file)
    print('saved accuracies to {}'.format(file_name))
    return file_name


def plot_accuracies(file_name):
    with open(file_name, 'rb') as file:
        a = pickle.load(file)
    plt.figure(figsize=(7, 4), tight_layout=True)
    e = a['epoch']
    plt.plot(e, a['train'], label='training')
    plt.plot(e, a['validate'], label='test')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('percent accuracy')
    title = 'file {}, {} weights'.format(file_name, a['weight count'])
    plt.title(title)
    plt.show()
    return a



if __name__ == '__main__':
    data_loaders = music_loaders('music.pkl')
    conv, fc = (2, 2), (100, 100)
    accuracies_file_name = experiment(data_loaders, conv, fc)
    accuracies = plot_accuracies(accuracies_file_name)
