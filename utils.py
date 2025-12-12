import gc
import torch
import matplotlib.pyplot as plt
import seaborn as sns


def cleanup(trainer=None, model=None, datamodule=None, logger=None):
    """
     cleanup the memory
    :param trainer:
    :param model:
    :param datamodule:
    :param logger:
    :return:
    """
    # Close logger
    if logger is not None:
        try:
            logger.experiment.flush()
            logger.experiment.close()
        except:
            pass

    # Clean Trainer workers
    try:
        if trainer is not None and trainer._data_connector:
            trainer._data_connector.teardown()
    except:
        pass

    # Delete major objects
    del trainer
    del model
    del datamodule

    # Python memory cleanup
    gc.collect()

    # CUDA cache cleanup
    torch.cuda.empty_cache()

    print(" Cleanup complete")


def plot_confusion_matrix(cm):
    """
    plot confusion matrix
    :param cm: confusion matrix
    :return:
    """
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("MNIST Confusion Matrix")
    plt.show()
