# Variational Autoencoder (VAE)

This repository contains a PyTorch implementation of a Variational Autoencoder (VAE) trained on the MNIST dataset of handwritten digits. It includes the model architecture, training loop, evaluation metrics, and comprehensive visualization to analyze the latent space and generative capabilities of the model.

---

## 1. Model Configuration

The VAE consists of a Convolutional Encoder and a Convolutional Transpose Decoder.

### Architecture Summary
* **Encoder:**
    * `Conv2d(1, 32, kernel_size=4, stride=2, padding=1)` $\rightarrow$ ReLU (Reduces spatial dimensions from 28x28 to 14x14)
    * `Conv2d(32, 64, kernel_size=4, stride=2, padding=1)` $\rightarrow$ ReLU (Reduces spatial dimensions from 14x14 to 7x7)
    * `Linear(64 * 7 * 7, 256)` $\rightarrow$ ReLU
    * `Linear(256, latent_dim)` for Latent Mean ($\mu$)
    * `Linear(256, latent_dim)` for Latent Log-Variance ($\log \sigma^2$)
* **Decoder:**
    * `Linear(latent_dim, 256)` $\rightarrow$ ReLU
    * `Linear(256, 64 * 7 * 7)` $\rightarrow$ ReLU
    * `ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)` $\rightarrow$ ReLU (Upsamples from 7x7 to 14x14)
    * `ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1)` $\rightarrow$ Sigmoid (Upsamples from 14x14 to 28x28)

### Hyperparameters
| Parameter | Value | Description |
| :--- | :--- | :--- |
| `latent_dim` | `20` | Size of the bottleneck/latent representation layer |
| `batch_size` | `128` | Number of training samples per optimization step |
| `learning_rate` | `1e-3` | Learning rate for the Adam optimizer |
| `epochs` | `10` | Total number of full passes over the training dataset |
| `optimizer` | `Adam` | Optimization algorithm used |
| `loss_function` | `BCE + KLD` | Binary Cross Entropy (reconstruction) + Kullback-Leibler Divergence |

---

## 2. Model Training Loss

The model is optimized using the Evidence Lower Bound (ELBO) objective, which minimizes the sum of two components:

$$\mathcal{L}_{\text{VAE}} = \mathcal{L}_{\text{Reconstruction}} + \mathcal{L}_{\text{KL}}$$

1.  **Reconstruction Loss ($\mathcal{L}_{\text{Reconstruction}}$):** Measures how well the decoded image matches the input image using Binary Cross Entropy (BCE).
2.  **KL Divergence ($\mathcal{L}_{\text{KL}}$):** Acts as a regularizer measuring how much the predicted latent distribution $q_\phi(z|x)$ deviates from a standard normal prior $p(z) = \mathcal{N}(0, I)$.

### Epoch and Loss

| Epoch | Loss |
| :---: | :---: |
| Epoch 1 | 183.5815 |
| Epoch 2 | 123.7691 |
| Epoch 3 | 112.1116 |
| Epoch 4 | 108.3040 |
| Epoch 5 | 106.3499 |
| Epoch 6 | 104.8317 |
| Epoch 7 | 103.7175 |
| Epoch 8 | 102.9064 |
| Epoch 9 | 102.2718 |
| Epoch 10 | 101.7468 |

---
## 3. Visualization

### 3.1 VAE Training Loss Curves
![[Alt]](assets/vae_training_loss_curve.png)

- Reconstruction loss decreases steadily and then stabilizes, showing that the encoder–decoder is successfully learning to compress and reconstruct images with increasing accuracy.

- KL divergence increases from near zero to a small stable value, indicating that the encoder gradually starts using the latent space while still keeping it close to a standard normal distribution. Early in training, the model often prioritizes reconstruction, leading to near-deterministic encodings. As training progresses, the KL term forces the latent distributions toward a standard normal prior, ensuring the latent space becomes structured and sampleable. The stabilization around a non-zero value is a sign of balanced regularization rather than collapse.

- Total loss decreases smoothly and converges, reflecting a stable balance between reconstruction quality and latent space regularization.

---

### 3.2 Original v/s Reconstruction

![[Alt]](assets/reconstruction.png)

---

### 3.3  Latent Space Visualization using PCA and TSNE

| Latent Space Visualization using PCA | Latent Space Visualization using TSNE |
| :---: | :---: |
| ![[Alt]](assets/pca.png) | ![[Alt]](assets/tsne.png) |

---

### 3.4 Latent Space Interpolation

![[Alt]](assets/interpolation.png) 



Latent space interpolation is a method used to understand how a Variational Autoencoder organizes and represents data internally. Instead of working directly with images in pixel space, a VAE compresses each image into a continuous vector in a learned latent space. Each point in this space represents a high-level, compressed version of the input image.

The main idea behind interpolation is to study what the model represents between two different inputs by moving smoothly between their latent representations and decoding each intermediate point back into an image.


#### **Mathematical Formulation**

Given two input images:

z1 = encoder(x1)  
z2 = encoder(x2)

We generate intermediate latent vectors using linear interpolation:

$$z(α) = (1 - α)z_{1} + α  z_{2}$$
where α ∈ [0, 1]

This creates a straight-line path in latent space between the two encoded points.

Behavior of the formula:
- α = 0 → z(α) = z1  
- α = 1 → z(α) = z2  
- 0 < α < 1 -> smooth transition between both  

Each latent vector is then decoded back into an image:

$$xhat(α) = decoder(z(α))$$

---

### 3.5 Random Sampling frmo latent Space

![[Alt]](assets/random.png) 

---

## 4. References and Credits:

This was possible becuase pf:

1. [Ahlad Kumar- VAE Playlist](https://www.youtube.com/playlist?list=PLdxQ7SoCLQANizknbIiHzL_hYjEaI-wUe)
2. [Umar Jamil - VAE explanation](https://youtu.be/iwEzwTTalbg?si=BO7wLM8oxqT2lYnP)

---


