import matplotlib.pyplot as plt


def plot_loss(history, save=False, filename=None):
    """
    :param history: tensorflow model
    :param save: OPTIONAL model saving
    :param filename: OPTIONAL saved model filename
    :return: None
    """
    plt.figure(1)
    plt.grid(True)
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    if save:
        if filename:
            plt.savefig(filename)
        else:
            raise ValueError("A filename is required to save the plot.")
    plt.close(1)


def plot_accuracy(history, save=False, filename=None):
    """
    :param history: tensorflow model
    :param save: OPTIONAL model saving
    :param filename: OPTIONAL saved model filename
    :return: None
    """
    plt.figure(2)
    plt.grid(True)
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim(bottom=0)
    plt.legend(loc='lower right')
    if save:
        if filename:
            plt.savefig(filename)
        else:
            raise ValueError("A filename is required to save the plot.")
    plt.close(2)
