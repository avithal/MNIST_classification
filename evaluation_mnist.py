import os
import yaml
import torch

from numpy import ceil
import matplotlib.pyplot as plt

from MNISTModel import MNISTModel
from MNISTDataModule import MNISTDataModule

with open('MNIST_simple.yaml', 'r') as f:
    config = yaml.safe_load(f)


def evaluate_and_find_misclassified_with_probs(model_path, loader):

    # model_to_load = MNISTModel(config['model'])
    model_pl = MNISTModel(config['model'])
    model_le = model_pl.model

    checkpoint = torch.load(model_path, map_location=torch.device("cpu"))
    state_dict = checkpoint["state_dict"]
    # Strip the "model." prefix from keys
    new_state_dict = {key.replace("model.", ""): value for key, value in state_dict.items()}

    # Load the modified state_dict into your model
    model_le.load_state_dict(new_state_dict)

    # Ensure the checkpoint matches the current model architecture
    # model_le.load_state_dict(checkpoint["state_dict"])

    # Step 4: Switch to evaluation mode
    model_le.eval()
    # Print the model architecture (optional)
    print(model_le)

    model_le.eval()
    misclassified = []
    count = 0
    with torch.no_grad():
        for images, labels in loader:
            count += 1
            outputs = model_le(images)
            probabilities = torch.softmax(outputs, dim=1)  # Convert logits to probabilities
            _, predicted = torch.max(outputs, 1)
            for i in range(len(labels)):
                if predicted[i] != labels[i]:
                    misclassified.append((images[i], labels[i].item(), predicted[i].item(), probabilities[i]))

    return misclassified


# Visualize misclassified samples with prediction probabilities
def visualize_misclassified_with_probs(misclassified, save_path, n=5):
    num_subplots = int(ceil(len(misclassified)/5))
    for j in range(num_subplots):
        plt.figure(figsize=(15, 5))
        for i in range(n):
            current_im = j*5+i
            if current_im > len(misclassified)-1:
                break
            image, true_label, predicted_label, probabilities = misclassified[current_im]
            image = image.squeeze().numpy()  # Convert to numpy for plotting

            # Plot the image
            plt.subplot(2, 5, i + 1)
            plt.imshow(image, cmap='gray')
            plt.title(f"True: {true_label}, Pred: {predicted_label}")
            plt.axis('off')

            # Plot the probabilities as a bar chart
            plt.subplot(2, 5, i + 1 + 5)
            plt.bar(range(10), probabilities.numpy())
            plt.xticks(range(10))
            plt.title("Prediction Probabilities")
            plt.xlabel("Classes")
            plt.ylabel("Probability")

            plt.tight_layout()
        plt.show(block=False)
        plt.savefig(save_path+"_"+str(j)+'.jpg', format='jpeg', dpi=300)  # High-quality image
        plt.close('all')


def total_images_loader(loader):
    total_samples = len(loader) * loader.batch_size
    last_batch_size = len(list(loader)[-1][0])
    total_samples = total_samples - loader.batch_size + last_batch_size
    return total_samples


if __name__ == "__main__":

    data_module = MNISTDataModule()
    data_module.setup_data()
    test_loader = data_module.test_dataloader()
    train_loader = data_module.test_dataloader()
    total_images = total_images_loader(train_loader)
    models_folder = r'D:\Avithal Study\MNIST_classification\DUMMY1'
    model_files = [os.path.join(models_folder, f) for f in os.listdir(models_folder)
                   if f.endswith(('.ckpt', '.pt', '.pth'))]

    if not model_files:
        raise ValueError("No model files found in the folder.")

    # Iterate through the models
    for model_file in model_files:

        # Run the updated evaluation
        misclassified_with_probs = evaluate_and_find_misclassified_with_probs(model_file, test_loader)
        print("misclassified_percentage[0-1]= ", len(misclassified_with_probs)/total_images)
        # Visualize the first 5 misclassified samples
        save_path_images = os.path.join(config['evaluation']['evaluation_dir'], model_file[:-5].split('\\')[-1])
        print(save_path_images)
        visualize_misclassified_with_probs(misclassified_with_probs, save_path_images)
