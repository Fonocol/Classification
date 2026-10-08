import matplotlib.pyplot as plt
import numpy as np


CIFAR10_CLASSES = [
    'avion', 'voiture', 'oiseau', 'chat', 'cerf',
    'chien', 'grenouille', 'cheval', 'bateau', 'camion'
]

def plot_sample_images(images, labels=None, class_names=None, num_rows=2, num_cols=5):

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 2, num_rows * 2))
    axes = np.array(axes).flatten()

    for idx, ax in enumerate(axes):
        if idx >= len(images):
            ax.axis('off')
            continue

        img = images[idx]

        # Si l'image est aplatie, la remettre en 2D pour l'affichage
        if img.ndim == 1:
            if img.size == 784:
                img = img.reshape(28, 28)
            elif img.size == 3072:
                img = img.reshape(32, 32, 3)


        if img.ndim == 2 or (img.ndim == 3 and img.shape[-1] == 1):
            ax.imshow(np.squeeze(img), cmap='gray')
        else:
            ax.imshow(img)

       
        if labels is not None:
            lbl = labels[idx]
            if isinstance(lbl, (np.ndarray, list)) and len(lbl) > 1:
                lbl = np.argmax(lbl)  #  si One-Hot
            title = class_names[lbl] if class_names else str(lbl)
            ax.set_title(title, fontsize=10)

        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_training_history(history):

    history_dict = history.history if hasattr(history, 'history') else history

    epochs = range(1, len(history_dict.get('loss', [])) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 1. Courbe de Loss
    axes[0].plot(epochs, history_dict.get('loss', []), label='Train Loss', color='blue')
    if 'val_loss' in history_dict:
        axes[0].plot(epochs, history_dict.get('val_loss', []), label='Val Loss', color='orange', linestyle='--')
    axes[0].set_title('Fonction de perte (Loss)')
    axes[0].set_xlabel('Époques')
    axes[0].set_ylabel('Perte')
    axes[0].legend()
    axes[0].grid(True)

    # 2. Courbe d'Accuracy
    acc_key = 'accuracy' if 'accuracy' in history_dict else 'acc'
    val_acc_key = 'val_accuracy' if 'val_accuracy' in history_dict else 'val_acc'

    if acc_key in history_dict:
        axes[1].plot(epochs, history_dict[acc_key], label='Train Accuracy', color='green')
    if val_acc_key in history_dict:
        axes[1].plot(epochs, history_dict[val_acc_key], label='Val Accuracy', color='red', linestyle='--')
    axes[1].set_title('Précision (Accuracy)')
    axes[1].set_xlabel('Époques')
    axes[1].set_ylabel('Précision')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()