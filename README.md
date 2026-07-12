# Emotion-Recognition with Deep Learning
Developed an AI-powered Real-Time Facial Emotion Recognition System capable of detecting and classifying human emotions from live webcam streams using Deep Learning and Computer Vision technologies.

This project performs **facial emotion recognition** using a **MobileNetV2-based Deep Learning model**. Two model versions are provided to enable direct performance comparison:

## **Model Versions**

### **1. Baseline Model (No Fine-Tuning)**

**MobileNetV2 backbone frozen**
**Validation Accuracy:** ~37%

### **2. Fine-Tuned Model**

* **Last 30 MobileNetV2 layers unfrozen**
* **Low learning rate** for careful optimization
* **Validation Accuracy:** ~40%

Both **`.keras` model files** are included in the repository.

---

## **Features**

* **Real-time emotion detection** using **OpenCV** and a webcam
* **Training code** for both baseline and fine-tuned models
* **Lightweight inference**, capable of running on laptops with integrated graphics
* **MobileNetV2 Transfer Learning** with **FER-style 48×48 grayscale images**

---

## **Training Workflow (Google Colab)**

Training was conducted on **Google Colab Free Tier** utilizing GPU acceleration.

The notebook includes:

* **Dataset preprocessing**
* Building the **baseline MobileNetV2 model**
* **Fine-tuning the last 30 layers**
* **Model evaluation**
* Saving trained models in **`.keras`** and **`.h5`** formats

You can easily retrain the model by opening the provided **`.ipynb` notebook** in **Google Colab**.

---

## **Real-Time Emotion Detection**

The real-time application loads the trained model and performs:

* **Face Detection** using **Haar Cascade Classifier**
* **Emotion Prediction** across **7 emotion classes**
* **Live Webcam Display** with predicted emotion labels

### **Clone the Repository**

```bash
git clone https://github.com/USERNAME/emotion-recognition.git
cd emotion-recognition
```

---

## **Environment Setup (pyenv + Python 3.11.6)**

This project was developed using **pyenv** with **Python 3.11.6**, as newer Python versions had compatibility limitations with **TensorFlow**.

Using **pyenv** provides a clean and isolated development environment, preventing dependency conflicts and ensuring reproducible installations.

### **1. Install pyenv**

Follow the official installation guide:

https://github.com/pyenv/pyenv

### **2. Install Python 3.11.6**

TensorFlow provides compatible CPU wheels for **Python 3.11**, allowing smooth installation within a managed virtual environment.

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Run Real-Time Emotion Detection**

```bash
python realtime_emotion.py
```

---

## **Libraries Used**

* **TensorFlow 2.15.0**
* **Keras 2.15.0**
* **OpenCV**
* **NumPy**
* **Matplotlib**
