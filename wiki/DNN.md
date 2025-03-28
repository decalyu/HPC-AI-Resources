# 🚀✨ **Workshop 4: 🤖 Mastering Deep Neural Networks (DNN)!** ✨🚀

---

# 🎯 **Goal**
🎉 Welcome to this hands-on workshop! You'll learn how to **build and train a Deep Neural Network (DNN)** from scratch using **TensorFlow** and **Keras**—even if you're new to coding! 🚀

---

# 📌 **What You Will Learn** 🧠💡
✅ What a **Deep Neural Network (DNN)** is and how it works 🤖  
✅ How to **build** a simple DNN model using TensorFlow & Keras 🏗️  
✅ How to **train** a neural network with real-world data 📊  
✅ How to **make predictions** with your trained model 🎯  
✅ How to visualize the **training progress** and results 📈  

---

# 🧠 **1. Understanding Deep Neural Networks** 🤔
### **Neural Networks = AI that Learns!** 💡
A **Deep Neural Network (DNN)** is a type of **machine learning model** inspired by the human brain. It consists of layers of **neurons**, which process and transform data step by step.

🧩 **Key Components of a DNN:**
- **Neuron**: A mathematical unit that processes inputs.
- **Layer**: A group of neurons working together.
- **Activation Function**: Helps the network learn complex patterns (e.g., **ReLU, Sigmoid**).

### 🖼️ **Visualizing a Simple Neural Network**
Below is an interactive visualization of a simple neural network. It demonstrates the structure of a network with an **input layer**, a **hidden layer**, and an **output layer**:

```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Neural Network Visualization</title>
    <style>
        body {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            background-color: #f4f4f4;
            font-family: Arial, sans-serif;
        }
        .container {
            position: relative;
            display: flex;
            justify-content: center;
            align-items: center;
        }
        .layer {
            display: flex;
            flex-direction: column;
            align-items: center;
            margin: 0 30px;
        }
        .neuron {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            background-color: #3498db;
            display: flex;
            justify-content: center;
            align-items: center;
            color: white;
            font-weight: bold;
            margin: 10px 0;
            box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
        }
        .label {
            text-align: center;
            font-size: 14px;
            margin-top: 5px;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="layer">
            <div class="label">Input Layer</div>
            <div class="neuron">X</div>
        </div>
        <div class="layer">
            <div class="label">Hidden Layer</div>
            <div class="neuron">N1</div>
            <div class="neuron">N2</div>
            <div class="neuron">N3</div>
        </div>
        <div class="layer">
            <div class="label">Output Layer</div>
            <div class="neuron">Y</div>
        </div>
    </div>
</body>
</html>
```

This visualization helps you understand how data moves through the network.

💡 **Quick Thought:** Where have you seen AI-powered predictions in daily life? 📝

---

# 🎬 **2. Hands-on: Building a DNN in Python** 💻
### 🚀 **Step 1: Open Google Colab**
1️⃣ Click **[Google Colab](https://colab.research.google.com/)**  
2️⃣ Click **New Notebook**  
3️⃣ 🎉 You’re ready to code!

### 📦 **Step 2: Install and Import Libraries**
```python
import tensorflow as tf  # Deep Learning Framework
from tensorflow import keras  # Simplifies Neural Networks
import numpy as np  # Handles Data
import matplotlib.pyplot as plt  # Visualizations
```
▶ Click **Run** to load the libraries.

---

# 📊 **3. Creating & Training a Simple DNN**
### **🔹 Prepare Data**
We will train a model where **output = input × 2**.
```python
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)  # Input
y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18, 20], dtype=float)  # Output
```
▶ Click **Run** to define the data.

### **🔹 Build the DNN Model**
```python
model = keras.Sequential([
    keras.layers.Dense(10, activation='relu', input_shape=(1,)),  # Hidden Layer
    keras.layers.Dense(1)  # Output Layer
])

model.compile(optimizer='sgd', loss='mse')  # Compile Model
```
▶ Click **Run** to create the model.

### **🔹 Train the Model**
```python
history = model.fit(X, y, epochs=100, verbose=0)  # Train for 100 epochs
```
▶ Click **Run** to train the model.

### **🔹 Visualize Training Progress**
```python
plt.plot(history.history['loss'])
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.show()
```
▶ Click **Run** to view training performance.

### **🔹 Make Predictions**
```python
prediction = model.predict([11])
print("Predicted Output for Input 11:", prediction[0][0])
```
▶ Click **Run** to see if the model predicts **22**.

---

# 🎯 **4. Wrap-Up & Next Steps**
🎉 **Congratulations!** You just built your first **Deep Neural Network!** Here’s what you accomplished:
✅ Built a **DNN model** from scratch 🤖  
✅ Trained it using real data 📊  
✅ Visualized learning progress 📈  
✅ Made predictions using AI 🏆  

🚀 **Next Workshop:** Exploring **real-world AI datasets** for more advanced applications! 🌍📊

---

# 🔗 **Additional AI Resources** 📚
🎉 Keep exploring AI, and see you at the next workshop! 🚀
