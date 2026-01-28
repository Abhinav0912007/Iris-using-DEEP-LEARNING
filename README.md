🌸 IRIS Flower Classification using Deep Learning (ANN)

This project implements an Artificial Neural Network (ANN) to classify iris flowers into three species based on their physical features. The model is built using TensorFlow/Keras and trained on the classic Iris dataset.

📌 Problem Statement

Given four flower features:

Sepal Length

Sepal Width

Petal Length

Petal Width

Predict the species of the Iris flower:

Iris-setosa

Iris-versicolor

Iris-virginica

🧠 Model Used: Artificial Neural Network (ANN)

The ANN is a feedforward neural network trained using supervised learning.

🔹 Architecture

Input Layer: 4 neurons

Hidden Layer 1: 16 neurons (ReLU)

Hidden Layer 2: 8 neurons (ReLU)

Output Layer: 3 neurons (Softmax)

🔹 Loss Function

Categorical Cross-Entropy

🔹 Optimizer

Adam

📂 Dataset Information

Dataset: Iris Dataset

Total Samples: 150

Classes: 3 (50 samples each)

Features: 4 numerical features

Target: Species (multiclass classification)

🛠️ Technologies & Libraries Used

Python

NumPy

Pandas

Matplotlib & Seaborn

Scikit-learn

TensorFlow / Keras

Streamlit (for UI, optional)

⚙️ Project Workflow

Data Loading

Exploratory Data Analysis (EDA)

Label Encoding of Target Variable

Feature Scaling using StandardScaler

Train-Test Split

ANN Model Building

Model Training

Model Evaluation

Visualization of Accuracy

Prediction using User Input

📈 Model Performance

Training Accuracy: ~98%

Testing Accuracy: ~93%

The ANN performs well on unseen data, showing strong generalization.

🧪 Evaluation Metrics

Accuracy

Loss

Validation Accuracy

Confusion Matrix (optional)

🚀 How to Run the Project
1️⃣ Install Dependencies
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow streamlit

2️⃣ Train the Model

Run the Jupyter Notebook to train and save the model.

3️⃣ Run Streamlit App (Optional)
streamlit run app.py

📁 Project Structure
iris-ann-project/
│
├── Iris.csv
├── app.py
├── iris_model.h5
├── scaler.pkl
├── iris_ann.ipynb
└── README.md

🎯 Applications

Educational ML/DL projects

Beginner deep learning classification problems

Model comparison (ANN vs traditional ML)

📜 License

This project is for educational purposes.

✨ Author

Developed by [Abhinav Singh]
Deep Learning | Machine Learning | Data Science
