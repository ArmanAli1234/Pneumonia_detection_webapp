# 🫁 PneumoScan AI — Pneumonia Detection Using DenseNet121

> An end-to-end Deep Learning project for detecting Pneumonia from Chest X-Ray images using **DenseNet121 Transfer Learning, Fine-Tuning, and Flask**.

## 📌 Overview

**PneumoScan AI** is a Deep Learning-based binary image classification system designed to classify chest X-ray images into:

* 🟢 **Normal**
* 🔴 **Pneumonia**

The project uses **DenseNet121 pretrained on ImageNet** as the backbone for feature extraction. Transfer learning is followed by fine-tuning to adapt the pretrained network to the chest X-ray pneumonia classification task.

The trained model is integrated with a **Flask web application**, allowing users to upload a chest X-ray image and receive a predicted class along with the model's probability score.

> ⚠️ This project is intended for educational and research purposes and is **not a medical diagnostic system**.

---

## ✨ Key Features

* DenseNet121 Transfer Learning
* ImageNet pretrained weights
* Fine-Tuning of pretrained layers
* Chest X-Ray image preprocessing
* 224 × 224 input resolution
* Binary classification using Sigmoid activation
* Configurable classification threshold
* Flask-based web application
* Image upload functionality
* Prediction probability display
* Real-time model inference
* Confusion Matrix and Classification Report evaluation
* ROC-AUC evaluation
* Focus on high Pneumonia sensitivity/recall

---

## 🧠 Model Architecture

The project uses **DenseNet121**, pretrained on ImageNet, as the feature extraction backbone.

### Model Pipeline

```text
Chest X-Ray Image
        ↓
Image Preprocessing
        ↓
Resize → 224 × 224
        ↓
DenseNet121
        ↓
Transfer Learning
        ↓
Fine-Tuning
        ↓
Dense(1, Sigmoid)
        ↓
Pneumonia Probability
        ↓
Threshold = 0.5
        ↓
Normal / Pneumonia
```

### Classification Head

The final classification layer uses a single neuron with Sigmoid activation:

```python
Dense(1, activation="sigmoid")
```

The model produces a probability between `0` and `1`.

```python
if prediction >= 0.5:
    result = "Pneumonia"
else:
    result = "Normal"
```

For a Normal prediction:

```python
normal_probability = 1 - prediction
```

---

## 🏗️ Project Architecture

```text
                    ┌────────────────────┐
                    │   User Uploads     │
                    │   Chest X-Ray      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │       Flask        │
                    │   Web Application  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Image Preprocessing│
                    │    Resize 224×224  │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │    DenseNet121     │
                    │   Trained Model    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │  Sigmoid Output    │
                    │   Probability      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ Normal / Pneumonia │
                    │   + Probability    │
                    └────────────────────┘
```

---

## 📂 Project Structure

```text
Pneumonia_detection/
│
├── app.py
├── prediction.py
├── requirements.txt
├── README.md
│
├── models/
│   └── pneumonia_model.keras
│
├── templates/
│   └── index.html
│
├── static/
│   ├── style.css
│   └── uploads/
│
└── Notebooks/
    └── Pneumonia_detection_project.ipynb
```

---

## 🛠️ Technologies Used

| Category             | Technologies               |
| -------------------- | -------------------------- |
| Programming Language | Python                     |
| Deep Learning        | TensorFlow, Keras          |
| CNN Architecture     | DenseNet121                |
| Transfer Learning    | ImageNet                   |
| Image Processing     | Pillow, NumPy              |
| Web Framework        | Flask                      |
| Frontend             | HTML, CSS                  |
| Template Engine      | Jinja2                     |
| Development          | VS Code, Jupyter Notebook  |
| Environment          | Python Virtual Environment |

---

# 📊 Model Performance

The model was evaluated on **624 samples of unseen test data**.

### Overall Performance

| Metric                |     Score |
| --------------------- | --------: |
| Accuracy              | **87.0%** |
| Precision — Pneumonia | **84.0%** |
| Recall — Pneumonia    | **97.0%** |
| F1-Score — Pneumonia  | **90.3%** |
| ROC-AUC               | **95.2%** |

### Detailed Classification Report

```text
              precision    recall    f1-score    support

Normal           0.94      0.70       0.80         234
Pneumonia        0.84      0.97       0.90         390

Accuracy                              0.87         624
Macro Avg        0.89      0.84       0.85         624
Weighted Avg     0.88      0.87       0.87         624
```

### Confusion Matrix

```text
                  Predicted
                Normal  Pneumonia

Actual Normal      164       70
Actual Pneumonia    11      379
```

### Key Observations

* **379 out of 390 Pneumonia cases were correctly detected.**
* Only **11 Pneumonia cases were classified as Normal**.
* Pneumonia recall reached approximately **97.2%**.
* The model achieved a **0.952 ROC-AUC**, indicating strong discrimination between Normal and Pneumonia cases.
* The model produces more false positives for Normal cases, which is reflected in the Normal recall of **70%**.

For this project, **recall/sensitivity for Pneumonia is particularly important**, because false-negative predictions represent Pneumonia cases that the model fails to detect.

---

## 🎯 Why Recall Matters

In medical screening-oriented classification, false negatives can be particularly important.

In this project:

```text
Actual Pneumonia
        ↓
Model Prediction
        ↓
379 Correctly Detected
11 Missed
```

Therefore:

```text
Pneumonia Recall
= TP / (TP + FN)

= 379 / (379 + 11)

≈ 97.2%
```

The model was therefore evaluated not only on overall accuracy, but also on **Pneumonia recall, precision, F1-score, and ROC-AUC**.

---

## 📈 ROC-AUC

The model achieved:

```text
ROC-AUC = 0.952
```

ROC-AUC measures how effectively the model separates the two classes across different classification thresholds.

A higher AUC indicates better class discrimination.

```text
0.50 → Random performance
0.70 → Fair
0.80 → Good
0.90+ → Excellent
1.00 → Perfect
```

The obtained **0.952 ROC-AUC** indicates strong overall discrimination between Normal and Pneumonia images.

---

## 🔬 Transfer Learning & Fine-Tuning

The project follows a two-stage training strategy.

```text
ImageNet Pretrained DenseNet121
                ↓
        Freeze Base Model
                ↓
       Train Classification Head
                ↓
          Unfreeze Layers
                ↓
           Fine-Tuning
                ↓
      Pneumonia Classification
```

### Why Transfer Learning?

Training a deep CNN completely from scratch requires a large amount of data and computational resources.

Using pretrained ImageNet weights allows DenseNet121 to start with previously learned visual representations and then adapt them to the chest X-ray classification task.

---

## 💡 Why DenseNet121?

DenseNet121 was selected as the backbone because of its dense connectivity between layers.

Each layer can receive feature information from preceding layers, which helps with:

* Feature reuse
* Gradient propagation
* Efficient parameter usage
* Learning deeper visual representations
* Transfer learning for image classification

This makes DenseNet121 a suitable architecture to experiment with for medical image classification tasks.

---

# 🖥️ Application Workflow

### Step 1 — Upload Image

The user uploads a chest X-ray image through the Flask interface.

### Step 2 — Image Preprocessing

The image is:

* Converted to RGB
* Resized to `224 × 224`
* Converted into a NumPy array
* Expanded to create the batch dimension

### Step 3 — Model Prediction

The processed image is passed to the trained DenseNet121 model.

```python
prediction = model.predict(image, verbose=0)[0][0]
```

### Step 4 — Classification

A threshold of `0.5` is applied:

```text
Probability >= 0.5 → Pneumonia
Probability < 0.5  → Normal
```

### Step 5 — Result

The Flask application displays:

* Prediction
* Probability/Confidence
* Uploaded X-ray
* Model information

---

## 📊 Example Prediction

```text
Prediction: Pneumonia
Probability: 85.57%

Model: DenseNet121
Input Size: 224 × 224
Task: Binary Classification
```

---

# 🌐 Flask Integration

The trained `.keras` model is integrated into a Flask web application.

The Flask backend handles:

* Image upload
* File validation
* Image preprocessing
* Model loading
* Model inference
* Prediction generation
* Result rendering

### Deployment Flow

```text
Browser
   ↓
Flask
   ↓
Image Upload
   ↓
Image Preprocessing
   ↓
DenseNet121
   ↓
Prediction
   ↓
HTML Result
```

---

## 🔐 Supported Image Formats

The application supports:

```text
.jpg
.jpeg
.png
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Pneumonia_detection
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

## 3. Activate Virtual Environment

### Windows

```powershell
venv\Scripts\activate
```

### Linux / macOS

```bash
source venv/bin/activate
```

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

The application will run locally at:

```text
http://127.0.0.1:5000
```

Open the URL in your browser and upload a chest X-ray image.

---

# 🚀 Future Improvements

* [ ] Add Grad-CAM visual explanations
* [ ] Experiment with classification thresholds
* [ ] Further reduce false positives
* [ ] Improve handling of class imbalance
* [ ] Add REST API endpoints
* [ ] Dockerize the application
* [ ] Add model monitoring
* [ ] Add application logging
* [ ] Optimize inference speed
* [ ] Add automated CI/CD
* [ ] Deploy using a production WSGI server
* [ ] Explore additional transfer learning architectures

---

# 📚 Key Concepts Demonstrated

* Convolutional Neural Networks
* DenseNet121
* Transfer Learning
* Fine-Tuning
* Image Classification
* Image Preprocessing
* Binary Classification
* Sigmoid Activation
* Classification Thresholds
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix
* TensorFlow/Keras
* Model Saving and Loading
* Flask
* Frontend-Backend Integration
* End-to-End ML Deployment

---

# ⚠️ Disclaimer

> **PneumoScan AI is developed strictly for educational, research, and demonstration purposes.**

This project is **not a medical diagnostic system** and should not be used for clinical decision-making. Chest X-ray interpretation should always be performed by qualified healthcare professionals in conjunction with appropriate clinical information.

---

# 👨‍💻 Author

## Arman Ali

**Aspiring Data Scientist | Machine Learning Enthusiast**

* GitHub: https://github.com/ArmanAli1234
* LinkedIn: https://www.linkedin.com/in/arman-ali-6b1169338/

---

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

### 🫁 PneumoScan AI

**DenseNet121 + Transfer Learning + Fine-Tuning + Flask**

> From Chest X-Ray image to AI prediction — an end-to-end Deep Learning project focused on high-sensitivity Pneumonia detection.
