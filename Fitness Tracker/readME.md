# 🏋️‍♂️ Smart Fitness Tracker using ML & Neural Networks

An intelligent fitness tracking system that leverages signal processing, machine learning, and deep learning to detect the **type of exercise**, **estimate effort intensity**, and **count repetitions** — all from motion sensor data (accelerometer & gyroscope).  

This system supports strength exercises like **deadlifts, squats, overhead press, bench press, and rows**.

---

## 📌 Features

- ✅ Detects exercise type (Deadlift, Squat, OHP, Bench, Row)
- ✅ Counts repetitions in real-time
- ✅ Classifies effort as **Light**, **Medium**, or **Heavy**
- ✅ Preprocesses motion sensor data using signal processing
- ✅ Extracts statistical and frequency-domain features
- ✅ Applies PCA for dimensionality reduction
- ✅ Trains multiple ML and DL classifiers
- ✅ Fine-tuned with GridSearchCV and Forward Feature Selection
- ✅ Neural Network trained with multiple epochs and adaptive learning

---

## 🚀 Demo

> Sample input: CSV file containing accelerometer and gyroscope data  
> Output:  
> - `exercise_type`: e.g., Squat  
> - `rep_count`: e.g., 10  
> - `effort`: e.g., Heavy

---

## 🧠 Algorithms Used

### Signal Processing
- **Butterworth Low-pass Filter** (noise reduction)
- **Fast Fourier Transform (FFT)** (frequency feature extraction)
- **Power Spectral Entropy** (signal complexity)

### Feature Engineering
- Rolling window features: mean, median, std, min, max
- Frequency-domain features: max freq, weighted freq
- PCA (Principal Component Analysis) for feature reduction

### Classification Models
- ✅ Decision Tree  
- ✅ Random Forest  
- ✅ Support Vector Machine (Linear & RBF)  
- ✅ K-Nearest Neighbors  
- ✅ Naive Bayes  
- ✅ Multi-layer Perceptron (Neural Network)

---

## 📂 Project Structure
```bash
fitness-tracker/
├── .vscode/ # VSCode settings
├── data/ # All data-related files
│ ├── interim/ # Intermediate preprocessed files
│ │ ├── 01_data_processed.pkl
│ │ ├── 02_outliers_removed_cahuv.pkl
│ │ └── 03_data_features.pkl
│ └── raw/ # Raw sensor data (CSV or similar)
├── reports/
│ └── figures/ # Generated plots, graphs, model visuals
│ └── extra/ # Extra visual assets or logs
├── src/ # Source code
│ ├── data/ # Dataset processing
│ │ ├── init.py
│ │ ├── first.ipynb
│ │ └── make_dataset.py
│ ├── features/ # Feature engineering and transformation
│ │ ├── init.py
│ │ ├── build_features.py
│ │ ├── count_repetetations.py
│ │ ├── DataTransformation.py
│ │ ├── FrequencyAbstraction.py
│ │ ├── remove_outliers.py
│ │ ├── TemporalAbstraction.py
│ │ └── second.ipynb
│ ├── models/ # ML models and training scripts
│ │ ├── pycache/
│ │ ├── LearningAlgorithms.py
│ │ └── train_model.py
│ └── visualization/ # Optional visualizations (currently empty)
├── environment.yml # Conda environment setup
├── readME.md # Project documentation
```


---

## 📊 Input Format

The system expects a CSV file containing raw motion sensor data:
There are more columns which are indroduced during the data analysis.. see through code

| timestamp | acc_x | acc_y | acc_z | gyro_x | gyro_y | gyro_z |
|-----------|-------|-------|-------|--------|--------|--------|
| 0.01      | 0.23  | -0.85 | 0.56  | 0.04   | 0.12   | -0.02  |
| 0.02      | 0.24  | -0.83 | 0.55  | 0.05   | 0.13   | -0.03  |
| ...       | ...   | ...   | ...   | ...    | ...    | ...    |

---

## 🏗️ How It Works

1. **Preprocessing**
   - Apply a low-pass filter to reduce high-frequency noise
   - Normalize and window the time-series signal
2. **Feature Extraction**
   - Compute rolling window stats and FFT-based features
   - Extract PSE and frequency-based weights
3. **Feature Selection & Reduction**
   - Use Forward Selection and PCA
4. **Model Training**
   - Train ML/DL models using extracted features
   - Tune hyperparameters using `GridSearchCV`
   - Train MLPClassifier using multiple epochs
5. **Prediction**
   - Classify the exercise type
   - Count repetitions based on repetition pattern segments
   - Predict effort level using movement intensity features

---

## 🧪 Example Output

```json
{
  "exercise_type": "Bench Press",
  "repetitions": 12,
  "effort": "Medium"
}
```

## 🧰 Tech Stack

- **Python 3.9+**
- **scikit-learn**
- **NumPy, pandas**
- **SciPy**
- **MLPClassifier, SVM, RandomForest**
- **Signal processing** (butter, fft, filtfilt)
- **Jupyter Notebook / Streamlit** (optional visualization)

---

## ✅ To Run the Project

### 1. Clone the repo

```bash
git clone https://github.com/gargdivyansh1/fitness-tracker
cd fitness-tracker
````

## 📈 Future Enhancements

- 📱 **Integrate with mobile apps (Android/iOS)** via Bluetooth sensors  
- ⏱️ **Real-time rep counting** using live sensor feed  
- 🤖 **Pose detection using video input** (OpenCV/MediaPipe)  
- ☁️ **Deploy model via Streamlit / Flask API**  
- 📊 **Dashboard for performance tracking and workout history**

---

## 🤝 Let's Connect

If you're working in **AI for fitness, healthcare, or wearable tech**, I'd love to collaborate or share insights!

**Divyansh Garg**  
🔗 [LinkedIn](https://www.linkedin.com/in/divyansh-garg515/)  
💻 [GitHub](https://github.com/gargdivyansh1)  
📩 divyanshgarg515@gmail.com
