# 🐱 Cat Breed Classification 
This repository contains two Jupyter notebooks for training a deep learning model that classifies **cat breeds** from images using **PyTorch**. The project loads a structured image dataset from Google Drive, applies standard computer-vision preprocessing, trains a CNN (with transfer learning), evaluates accuracy, and saves the trained model for reuse.

---

## 📁 Files in This Repository

* **`LAB1_GENAI.ipynb`** – Main notebook used for dataset loading, preprocessing, training, validation, and saving the model.
* **`Custom_Cnn.ipynb`** – Notebook with a custom CNN architecture and experiments.

Both notebooks follow the same data pipeline and produce comparable results.

---

## 📂 Dataset (Google Drive)

The dataset used in this project is stored in Google Drive and organized by breed folders (compatible with PyTorch `ImageFolder`).

🔗 **Google Drive Dataset Link:**
[https://drive.google.com/drive/folders/1UXNli9LovIuu6LheKiG9jTa_ECPoytfI?usp=drive_link](https://drive.google.com/drive/folders/1UXNli9LovIuu6LheKiG9jTa_ECPoytfI?usp=drive_link)

### 📊 Structure

```
cat_datasets/
├── Abyssinian/
├── Bengal/
├── Persian/
├── Siamese/
├── Maine Coon/
├── Ragdoll/
├── Russian Blue/
└── ... (other cat breeds)
```

Each folder contains images of a single breed. This structure allows direct loading using `torchvision.datasets.ImageFolder`.

---

## 🧠 Data Preprocessing

Both notebooks apply the following transformations before training:

```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

* Images are resized to **224×224**
* Converted to tensors
* Normalized using **ImageNet statistics** for better transfer-learning performance

---

## ⚙️ Model & Training

### Framework

* **PyTorch**

### Approach

* Dataset loaded using `ImageFolder`
* Split into **training** and **validation** subsets
* **Transfer Learning** with a pretrained CNN (e.g., ResNet) by replacing the final classification layer to match the number of cat breeds
* **Loss Function:** CrossEntropyLoss
* **Optimizer:** Adam

### Training

* Typical training over multiple epochs
* Accuracy evaluated on a validation set after each epoch

---

## 💾 Model Saving

To prevent data loss in Google Colab, the trained model is saved directly to Google Drive:

```python
model_path = "/content/drive/MyDrive/cat_species_model.pth"
torch.save(model.state_dict(), model_path)
```

The model can later be reloaded for evaluation or inference:

```python
model.load_state_dict(torch.load(model_path))
model.eval()
```

---

## ▶️ How to Run

1. Open either notebook (`LAB1_GENAI.ipynb` or `Custom_Cnn.ipynb`) in **Google Colab**.
2. Mount Google Drive:

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```
3. Ensure the dataset is available at:

   ```
   /content/drive/MyDrive/cat_datasets
   ```
4. Run all cells to train the model.
5. The trained model will be saved to your Drive for future use.

---

## 📈 Results

The model achieves strong validation accuracy across multiple cat breeds using transfer learning and standardized preprocessing. Performance may vary depending on the number of epochs and dataset size, but results are consistent across both notebooks.

*(You can add your exact accuracy here if required for reporting.)*

## 👨‍💻 Author

**Nitesh Goel**
B.Tech CSE | Generative AI / Deep Learning Project
