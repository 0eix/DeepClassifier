import torch
from torch import nn
from torchvision.models import resnet50, alexnet, vgg13, vgg16, densenet161
from torchvision.models import ResNet50_Weights, AlexNet_Weights, VGG13_Weights, VGG16_Weights, DenseNet161_Weights


def freeze_model_params_(model):
    """
    Freeze all the parameters of a model.

    Args:
        model (nn.Module): a convolutional neural network

    Returns:
        None
    """
    for param in model.parameters():
        param.requires_grad = False


def get_model_trainable_params(model):
    """
    Get the unfreezed params in a model.

    Args:
        model (nn.Module): a convolutional neural network

    Returns:
        trainable_params (List): list of parameters to learn during the
        training.
    """
    return filter(lambda param: param.requires_grad, model.parameters())


def build_classifier(num_features, num_hidden_units, num_classes, dropout):
    """
    Build a classifier.

    Args:
        num_features (Int): the number of units in the input layer

        num_hidden_units (Int): the number of units in the hidden layer

        num_classes (Int): the number of units in the output layer

        dropout (Float): the nodes drop probability

    Returns:
        classifier (nn.Module): a model capable of classification
    """
    classifier = nn.Sequential(
        nn.Linear(num_features, num_hidden_units),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(num_hidden_units, num_classes),
        nn.LogSoftmax(dim=1),
    )

    return classifier


#  Inspired by pytorch documentation.
# SEE:
# https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
def build_model(arch, num_hidden_units, num_classes, dropout):
    """
    Download the appropriate pretrained model according to the specified
    architecture and adjust the classifier.

    Args:
        arch (String): the architecture of the pretrained models.

        num_hidden_units (Unt): the number of units in the hidden layer of the
        classifier.

        num_classes (Int): the number of classes in the dataset.

        dropout (Float): the nodes drop probability

    Returns:
        model (nn.Module): a convolutional neural network with a classifier
        layer ready to be trained.
    """

    if arch == "resnet50":
        model = resnet50(weights=ResNet50_Weights.DEFAULT)
        freeze_model_params_(model)
        num_features = model.fc.in_features
        model.fc = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "alexnet":
        model = alexnet(weights=AlexNet_Weights.DEFAULT)
        freeze_model_params_(model)
        num_features = model.classifier[0].in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "vgg13":
        model = vgg13(weights=VGG13_Weights.DEFAULT)
        freeze_model_params_(model)
        num_features = model.classifier[0].in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "vgg16":
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        freeze_model_params_(model)
        num_features = model.classifier[0].in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    elif arch == "densenet161":
        model = densenet161(weights=DenseNet161_Weights.DEFAULT)
        freeze_model_params_(model)
        num_features = model.classifier.in_features
        model.classifier = build_classifier(
            num_features, num_hidden_units, num_classes, dropout
        )

    else:
        print("Invalid model name, exiting...")
        exit()

    return model


def rebuild_model_from_checkpoint(checkpoint_path):
    """
    Rebuild a model from its checkpoint.

    Args:
        checkpoint_path (Path): a path to the model's checkpoint.

    Returns:
        model (nn.Module): a convolutional neural network with a classifier
        layer ready to be retrained of inferred.
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = build_model(
        checkpoint["arch"],
        checkpoint["num_hidden_units"],
        checkpoint["num_classes"],
        checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.class_to_idx = checkpoint["class_to_idx"]

    return model
