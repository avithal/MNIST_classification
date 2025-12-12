from torch import nn


def simple_lenet(num_classes=10):
    """
    simple Lenet
    :param num_classes:  number of classes
    :return: model
    """
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),  # 28×28 → 24×24
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),  # 24×24 → 12×12

        nn.Conv2d(6, 16, kernel_size=5),  # 12×12 → 8×8
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),  # 8×8 → 4×4

        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        nn.ReLU(inplace=True),
        nn.Linear(120, 84),
        nn.ReLU(inplace=True),
        nn.Linear(84, num_classes)
    )
    return model


def lenet_batchnorm(num_classes=10):
    """
    batchnorm  Lenet
    :param num_classes:  number of classes
    :return: model
    """
    model = nn.Sequential(
        nn.Conv2d(1, 6, kernel_size=5),  # 28×28 → 24×24
        nn.BatchNorm2d(6),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),  # 24×24 → 12×12

        nn.Conv2d(6, 16, kernel_size=5),  # 12×12 → 8×8
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.AvgPool2d(2, 2),  # 8×8 → 4×4

        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        nn.BatchNorm1d(120),
        nn.ReLU(inplace=True),
        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.ReLU(inplace=True),
        nn.Linear(84, num_classes)
    )
    return model


def lenet_droput(num_classes=10):
    """
    dropout Lenet
    :param num_classes:  number of classes
    :return: model
    """


def lenet_batchnorm_dropout(num_classes=10, dropout_fc=0.5, dropout_conv=0.2):
    model = nn.Sequential(
        # Conv block 1
        nn.Conv2d(1, 6, kernel_size=5),
        nn.BatchNorm2d(6),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout_conv),  # Drop some channels
        nn.AvgPool2d(2, 2),

        # Conv block 2
        nn.Conv2d(6, 16, kernel_size=5),
        nn.BatchNorm2d(16),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout_conv),
        nn.AvgPool2d(2, 2),

        # Fully connected
        nn.Flatten(),
        nn.Linear(16 * 4 * 4, 120),
        nn.BatchNorm1d(120),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_fc),  # Drop neurons in FC

        nn.Linear(120, 84),
        nn.BatchNorm1d(84),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout_fc),

        nn.Linear(84, num_classes)
    )
    return model


def multi_layer_perceptron(num_classes=10):
    """
    sinple MLP
    :param num_classes:
    :return:
     model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, num_classes),
    )
    return model
