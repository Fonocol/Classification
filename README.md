# Classification d'Images : MLP vs CNN (LeNet-5)

Projet de comparaison entre deux architectures de réseaux de neurones (**Perceptron Multicouche** et **Convolutional Neural Network - LeNet-5**) appliquées aux jeux de données **MNIST** et **CIFAR-10**.

Le projet inclut l'entraînement, l'évaluation et une application web interactive Streamlit permettant de tester les modèles en temps réel.

---

## Architecture du Projet

```text
CLASSIFICATION/
│
├── utils/                          # Modules utilitaires réutilisables
│   ├── data_loader.py              # Chargement, normalisation et encodage One-Hot
│   ├── models.py                   # Définitions des architectures (MLP & LeNet-5)
│   └── plots.py                    # Visualisation des données et courbes d'entraînement
│
├── notebook_mlp_mnist.ipynb        # Entraînement et évaluation du MLP sur MNIST
├── notebook_cnn_mnist.ipynb        # Entraînement et évaluation de LeNet-5 sur MNIST
├── notebook_mlp_cifar10.ipynb      # Entraînement et évaluation du MLP sur CIFAR-10
├── notebook_cnn_cifar10.ipynb      # Entraînement et évaluation de LeNet-5 sur CIFAR-10
│
├── mlp_mnist.keras                 # Modèle entraîné MLP pour MNIST
├── cnn_mnist.keras                 # Modèle entraîné LeNet-5 pour MNIST
├── mlp_cifar.keras                 # Modèle entraîné MLP pour CIFAR-10
├── cnn_cifar.keras                 # Modèle entraîné LeNet-5 pour CIFAR-10
│
├── app.py                          # Interface interactive Streamlit
└── requirements.txt                # Dépendances Python du projet