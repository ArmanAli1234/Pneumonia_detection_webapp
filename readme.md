# 🫁 PneumoScan AI — Pneumonia Detection Using EfficientNetB2

> An end-to-end Deep Learning project for detecting Pneumonia from Chest X-Ray images using EfficientNetB2 Transfer Learning and Flask deployment.

## 📌 Overview

PneumoScan AI is a Deep Learning-based binary image classification system that classifies chest X-ray images into:

* 🟢 Normal
* 🔴 Pneumonia

The project uses **EfficientNetB2 with Transfer Learning**, followed by fine-tuning for the pneumonia classification task.

The trained model is deployed using **Flask**, allowing users to upload a chest X-ray image through a web interface and receive a prediction along with the model's probability score.

## ✨ Features

* EfficientNetB2 Transfer Learning
* Fine-Tuning of pretrained layers
* Chest X-Ray image preprocessing
* 224 × 224 input images
* Binary classification using Sigmoid activation
* 0.5 classification threshold
* Flask web application
* Image upload functionality
* Prediction probability display
* Modern responsive UI
* Real-time model inference

## 🧠 Model Architecture

The project uses **EfficientNetB2**, pretrained on ImageNet, as the backbone for feature extraction.

### Model Pipeline

```text
Chest X-Ray
     ↓
Image Preprocessing
     ↓
Resize → 224 × 224
     ↓
EfficientNetB2
     ↓
Transfer Learning
     ↓
Fine-Tuning
     ↓
Dense(1, Sigmoid)
     ↓
Probability
     ↓
Threshold = 0.5
     ↓
Normal / Pneumonia
```

## 🔬 Classification Logic

The final layer uses a single neuron with **Sigmoid activation**:

```python
Dense(1, activation="sigmoid")
```

The model outputs a value between `0` and `1`.

```python
if prediction >= 0.5:
    result = "Pneumonia"
else:
    result = "Normal"
```

### Example

```text
Model Output: 0.8557

0.8557 >= 0.5

Prediction: Pneumonia
Probability: 85.57%
```

For a Normal prediction:

```python
normal_probability = 1 - prediction
```

## 🏗️ Project Architecture

```text
                    ┌───────────────────┐
                    │    User Uploads   │
                    │     X-Ray Image   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │      Flask        │
                    │   Web Application │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Image Processing  │
                    │   Resize 224×224  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │   EfficientNetB2  │
                    │   Trained Model   │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Sigmoid Output    │
                    │   Probability     │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Normal / Pneumonia│
                    │ + Confidence      │
                    └───────────────────┘
```

## 📂 Project Structure

```text
Pneomonia_detection/
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
    └── Pneomonia_detection_project.ipynb
```

## 🛠️ Technologies Used

| Category          | Technologies                 |
| ----------------- | ---------------------------- |
| Language          | Python                       |
| Deep Learning     | TensorFlow, Keras            |
| Architecture      | EfficientNetB2               |
| Transfer Learning | ImageNet                     |
| Image Processing  | Pillow, NumPy                |
| Web Framework     | Flask                        |
| Frontend          | HTML, CSS                    |
| Template Engine   | Jinja2                       |
| Development       | VS Code, Virtual Environment |

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Pneomonia_detection
```

### 2. Create Virtual Environment

```bash
python -m venv venv
```

### 3. Activate Virtual Environment

#### Windows

```powershell
venv\Scripts\activate
```

#### Linux / macOS

```bash
source venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

## ▶️ Run the Application

Start the Flask server:

```bash
python app.py
```

The application will run at:

```text
http://127.0.0.1:5000
```

Open the URL in your browser and upload a chest X-ray image.

## 🖥️ How It Works

### Step 1 — Upload Image

The user uploads a chest X-ray image through the web interface.

### Step 2 — Image Preprocessing

The uploaded image is:

* Converted to RGB
* Resized to `224 × 224`
* Converted into a NumPy array
* Expanded to create the batch dimension

### Step 3 — Model Prediction

The processed image is passed to the trained EfficientNetB2 model.

```python
prediction = model.predict(image, verbose=0)[0][0]
```

### Step 4 — Classification

A threshold of `0.5` is applied.

```text
Probability >= 0.5 → Pneumonia
Probability < 0.5  → Normal
```

### Step 5 — Result

The Flask application displays:

* Prediction
* Confidence/Probability
* Uploaded X-ray
* Model information

## 📊 Example Prediction

```text
Prediction: Pneumonia
Probability: 0.8557
```

The application displays:

```text
Prediction
Pneumonia Detected

Model Confidence
85.57%

Model
EfficientNetB2

Input Size
224 × 224

Task
Binary Classification
```

## 📈 Model Evaluation

The model should be evaluated using:

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Confusion Matrix

For pneumonia classification, **Recall is an important metric** because false-negative predictions can be particularly important in screening scenarios.

Add the final test-set metrics here:

```text
Accuracy : XX.XX%
Precision: XX.XX%
Recall   : XX.XX%
F1-Score : XX.XX%
ROC-AUC  : XX.XX%
```

## 💡 Why EfficientNetB2?

EfficientNetB2 was selected because it provides a good balance between:

* Model capacity
* Computational efficiency
* Feature extraction capability
* Transfer learning performance

Using ImageNet pretrained weights allows the model to start with learned visual features instead of learning everything from scratch.

## 🔄 Transfer Learning & Fine-Tuning

The project follows a two-stage approach:

```text
ImageNet Pretrained EfficientNetB2
                ↓
        Freeze Base Model
                ↓
       Train Classification Head
                ↓
          Fine-Tuning
                ↓
      Pneumonia Classification
```

## 🌐 Flask Deployment

The trained `.keras` model is integrated into a Flask application.

The Flask backend handles:

* Image upload
* File validation
* Image preprocessing
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
Preprocessing
   ↓
EfficientNetB2
   ↓
Prediction
   ↓
HTML Result
```

## 🔐 Supported Image Formats

The application supports:

```text
.jpg
.jpeg
.png
```

## 🚀 Future Improvements

* [ ] Deploy on AWS / Azure / GCP
* [ ] Dockerize the application
* [ ] Add REST API endpoints
* [ ] Add Grad-CAM visual explanations
* [ ] Add model monitoring
* [ ] Add logging
* [ ] Optimize inference speed
* [ ] Experiment with classification thresholds
* [ ] Add automated CI/CD
* [ ] Add production WSGI server
* [ ] Improve handling of class imbalance

## 📚 Key Concepts Demonstrated

* Convolutional Neural Networks
* Transfer Learning
* EfficientNetB2
* Fine-Tuning
* Image preprocessing
* Binary classification
* Sigmoid activation
* Classification thresholds
* Model saving/loading
* TensorFlow/Keras inference
* Flask deployment
* Frontend-backend integration
* End-to-end ML deployment

## ⚠️ Disclaimer

> **This project is developed for educational, research, and demonstration purposes only.**

It is **not a medical diagnostic system** and should not be used for clinical decision-making or as a replacement for evaluation by a qualified healthcare professional.

## 👨‍💻 Author

### Arman Ali

**Aspiring Data Scientist | Machine Learning Enthusiast**

* GitHub: https://github.com/ArmanAli1234
* LinkedIn: https://www.linkedin.com/in/arman-ali-6b1169338/

## ⭐ Support

If you found this project useful, consider giving the repository a ⭐ on GitHub.

---

### 🫁 PneumoScan AI

**EfficientNetB2 + Transfer Learning + Flask**

> From X-Ray image to AI prediction — an end-to-end Deep Learning deployment project.
