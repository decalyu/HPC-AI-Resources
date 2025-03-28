# 🚀✨ PC: Deep Neural Network (DNN) ✨

## 🎯 Goal

📊 Learn how to **build, train, and visualize** a Deep Neural Network (DNN) using TensorFlow and Keras. No prior experience needed—just bring your curiosity and creativity! 🚀

---
## 📌 What You Will Learn 🧠💡

✅ What **datasets** are and why they matter 📂<br>
✅ How to **prepare and process** real-world data 🔍<br>
✅ How to **build** a Deep Neural Network 🏗️<br>
✅ How to **train and evaluate** the DNN 🔄<br>
✅ How to **visualize** training results 📈<br>
✅ Hands-on coding with **Google Colab** for easy Python use 💻<br>

---
## 📚 1. Key Terminologies in Deep Neural Networks 🧠

### 🧠 What is a Neuron?

A neuron (also called a node or unit) is the basic building block of a neural network, similar to how a neuron in the human brain processes information. It takes in inputs, performs a mathematical operation, and then produces an output.

### 🖼️ Deep Neural Network Visual Representation

<img width="600" alt="Screenshot 2025-03-11 at 3 33 33 PM" src="https://github.com/user-attachments/assets/4bdbeb87-510b-40b2-b3ff-5c4d446146eb" />

Source: [Google Images](https://medium.com/data-science-365/overview-of-a-neural-networks-learning-process-61690a502fa)

### 🧠 How a Neural Network Learns 🔗 
 
💡 Think: How do these concepts work together to train an artificial intelligence (AI) model? 🤔

Neural networks are a fundamental part of AI. They learn by adjusting their internal settings based on data. This process involves two main steps: **forward propagation** (making predictions) and **backward propagation** (learning from mistakes). 

- **Neurons** 🧠    
  - Represented by colored circles (orange, blue, green, and purple).
Each neuron processes inputs, applies an activation function, and passes the output to the next layer.

- **Layers** 🏗️
  - Input Layer: The orange circles on the left take input features (e.g., x1, x2).
  - Hidden Layers: The blue and green circles process the information between input and output.
  - Output Layer: The purple circle at the end provides the final prediction.

- **Activation Function** ⚡
  - Indicated inside the blue, green, and purple neurons as z, f.
  - Each neuron applies an activation function (e.g., ReLU, Sigmoid) to determine whether the signal should pass through.

- **Weights & Biases** ⚖️
  - Weights are represented by the solid black arrows connecting neurons.
  - Bias terms are represented by the gray circles, which are adjusted during training to improve model accuracy.

- **Forward Propagation** ➡️
  - Represented by the red arrow at the top.
  - The process of passing inputs (x1, x2) through the network to generate Predictions (y').
  - Information flows from left to right in the network.

- **Loss Function** ❌
  - Represented by the blue oval labeled "Loss Function".
  - It calculates the difference between the Predictions (y') and True Values (y) to measure model accuracy.
  - Examples include Mean Squared Error (MSE) and Cross-Entropy Loss.

- **Optimizer** 🔄
  - Represented by the red oval labeled "Optimizer".
  - It updates Weights using techniques like Stochastic Gradient Descent (SGD) or Adam to minimize the Loss Function.

- **Backpropagation** 🔙
  - Represented by the red arrow at the bottom.
  - It is the process of adjusting weights and biases by moving backward from the Loss Function to the Optimizer.
  - Ensures that the model learns by reducing prediction errors iteratively.

- **Epochs & Batch Size** 🔁
  - The blue circular arrows indicate the iterative training process.
  - Epochs: The number of times the entire dataset is processed.
  - Batch Size: The amount of data used in one training step before updating weights.

---


### 🚀 Step 1: Forward Propagation (Making Predictions)

Forward propagation is the process of passing inputs through the network to generate predictions.

### **How It Works**
- **Input Layer**: The network receives raw data (e.g., temperature, humidity).
- **Hidden Layers**:
   - Each neuron applies a **weighted sum** to the inputs.
   - The result passes through an **activation function** that helps decide which information is important.
- **Output Layer**: The final result (prediction) is generated.

### **Example**

If we are predicting whether it will rain:

- **Input**: Humidity = 80%, Temperature = 25°C
- **Hidden Layer Processing**: Mathematical operations adjust these numbers.
- **Output**: "Yes, it will rain" (Prediction).

### 🚀 Step 2: Loss Function (Measuring Mistakes)

The prediction is compared to the actual answer. The **loss function** calculates how far off the prediction was:
- A **low loss** means the prediction was close.
- A **high loss** means the network needs improvement.
- This matters because the model uses the loss to learn and get better over time.
- The goal is to reduce the loss so the model makes more accurate predictions.

### 🚀 Step 3: Backward Propagation (Learning from Mistakes)

Backward propagation adjusts the network to reduce mistakes.

### **How It Works**

- **Error Calculation**: The loss function determines how wrong the prediction was.
- **Weight Adjustment**:
   - The **optimizer** updates the weights using **gradient descent** (a technique to find the best values).
- **Repeat Until Accurate**: The process repeats multiple times, gradually improving accuracy.

### 🚀 Step 4: Why This Matters

Neural networks power many AI applications which are used in everyday life to make intelligent predictions

- 👩🏻‍⚕️ **Healthcare** → Predicts diseases from medical images  
- 🚗 **Self-Driving Cars** → Identifies pedestrians, traffic signs, and lanes  
- 📈 **Finance** → Predicts stock market trends and detects fraudulent transactions  
- 📺 **Entertainment** → Recommends movies and music based on your preferences  
- 📱 **Language Processing** → Powers virtual assistants like Siri or Google Assistant  

By using forward and backward propagation, AI **learns and improves over time**, just like humans learning from experience.

---

## 🔍 2. Hands-on: Exploring a Real Dataset

### Open **[Google Colab](https://colab.research.google.com/)**

1️⃣ Open your browser and go to Google Colab.<br>
2️⃣ Click **+ New notebook**.<br>

---

## 💾 3. Loading the Iris Dataset

### **➕🐍 Add a New Code Cell**  

1️⃣ Click **+ Code** in the top left to add a new code cell.<br>
2️⃣ Copy and paste the following code into the new code cell.<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e3f769-9cc4-8008-94dd-4603ea46574b)

```python
# Import pandas for handling data
import pandas as pd 

# Import sample data
from sklearn.datasets import load_iris

# Load the Iris dataset
overall_dataset = load_iris()

# Load the dataset into a DataFrame
data = pd.DataFrame(overall_dataset.data, columns=overall_dataset.feature_names)

# Add species (target labels) to the dataset
data['species'] = overall_dataset.target

print("Dataset loaded successfully! 🎉")
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e3f769-9cc4-8008-94dd-4603ea46574b)

3️⃣ **Click Run (▶) and check the output!** 

✅ Dataset loaded successfully! You should now see a message confirming the data is ready. 🎉

---
## 🏗️ 4. Building the DNN Model

### **➕🐍 Add a New Code Cell**  

1️⃣ Click **+ Code** in the top left to add a new code cell.<br>
2️⃣ Copy and paste the following code into the new code cell.<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e3f7fa-a7a4-8008-8a1c-9b7574327198)

```python
# Import function to split data into training and test sets
from sklearn.model_selection import train_test_split

# Import OneHotEncoder to convert labels into numerical form
from sklearn.preprocessing import OneHotEncoder

# Import PyTorch for building neural networks
import torch

# Import PyTorch's neural network module
import torch.nn as nn

# Import PyTorch's optimizer module
import torch.optim as optim

# Import utilities for handling data in PyTorch
from torch.utils.data import DataLoader, TensorDataset

# Split data into training and test sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(overall_dataset.data, overall_dataset.target, test_size=0.2, random_state=42)

# Convert labels into a format the model can understand (one-hot encoding)
ohe = OneHotEncoder(sparse_output=False)
y_train_encoded = ohe.fit_transform(y_train.reshape(-1, 1))
y_test_encoded = ohe.transform(y_test.reshape(-1, 1))

# Convert data into PyTorch tensors (which are like NumPy arrays but work with PyTorch)
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train_encoded, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test_encoded, dtype=torch.float32)

# Create datasets for PyTorch and enable batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# Define a simple neural network with three layers
class DNN(nn.Module):
    def __init__(self):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(4, 10)  # First layer (input: 4 features, output: 10 neurons)
        self.layer2 = nn.Linear(10, 10)  # Second layer (input: 10 neurons, output: 10 neurons)
        self.layer3 = nn.Linear(10, 3)  # Third layer (input: 10 neurons, output: 3 classes)
        self.relu = nn.ReLU()  # Activation function to introduce non-linearity
        self.softmax = nn.Softmax(dim=1)  # Softmax to convert output into probabilities

    def forward(self, x):  # Define how data moves through the network
        x = self.relu(self.layer1(x))  # Apply first layer and ReLU activation
        x = self.relu(self.layer2(x))  # Apply second layer and ReLU activation
        x = self.softmax(self.layer3(x))  # Apply third layer and softmax activation
        return x

# Create the model
model = DNN()
print("DNN model created successfully! 🎉")

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e3f7fa-a7a4-8008-8a1c-9b7574327198)

3️⃣ **Click Run (▶) and check the output!** 

✅ DNN model created successfully! Your neural network is now ready for training. 🎉

---
## 🎯 5. Training the DNN Model

### **➕🐍 Add a New Code Cell**  

1️⃣ Click **+ Code** in the top left to add a new code cell.<br>
2️⃣ Copy and paste the following code into the new code cell.<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e3f85f-d1fc-8008-9c78-8b7bba0d3787)

```python
# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
# Optimizer to adjust weights to minimize loss
optimizer = optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 50  # Number of times the model sees the entire dataset
train_losses = []  # Store loss values for plotting
for epoch in range(epochs):
    epoch_loss = 0.0  # Track loss for the epoch
    for X_batch, y_batch in train_loader:  # Loop through data in batches
        optimizer.zero_grad()  # Reset gradients before each update
        outputs = model(X_batch)  # Get model predictions
        loss = criterion(outputs, y_batch)  # Calculate loss
        loss.backward()  # Compute gradients
        optimizer.step()  # Update model weights
        epoch_loss += loss.item()  # Accumulate loss
    train_losses.append(epoch_loss / len(train_loader))  # Store average loss for this epoch
    print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss / len(train_loader):.4f}")  # Print progress

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e3f85f-d1fc-8008-9c78-8b7bba0d3787)

3️⃣ **Click Run (▶) and check the output!** 

✅ Model training started! You should see the training progress with loss and accuracy metrics. 🎉

---

## 📈 6. Visualizing Training Progress

### **➕🐍 Add a New Code Cell**  

1️⃣ Click **+ Code** in the top left to add a new code cell.<br>
2️⃣ Copy and paste the following code into the new code cell.<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e3f8a0-8ce8-8008-9655-b67415a10ee2)

```python
# Import Matplotlib for plotting graphs
import matplotlib.pyplot as plt

# Plot training loss
plt.plot(train_losses)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Progress')
plt.show()

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e3f8a0-8ce8-8008-9655-b67415a10ee2)

3️⃣ **Click Run (▶) and check the output!** 

✅ Training progress visualized! You should now see a loss curve showing how the model's performance changes over epochs. 📈🎉

The following graph shows the **loss (error) reduction** during training. The loss function helps us understand how well the Deep Neural Network (DNN) is learning over time.

<img src="https://github.com/user-attachments/assets/cb39433e-788d-44e3-bfa7-a5b6af02a496" alt="Training Progress - Loss Reduction" width="450">

- The **X-axis (Epochs)** represents the number of training iterations, and the **Y-axis (Loss)** represents the remaining error in the model’s predictions. 

- The decreasing trend indicates that the model is learning and improving!

---

## 🏆 7. Evaluating the Model

### **➕🐍 Add a New Code Cell**          
1️⃣ Click **+ Code** in the top left to add a new code cell.<br>
2️⃣ Copy and paste the following code into the new code cell.<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e3f914-ea3c-8008-a0d2-464ef8e20ece)

```python
# Evaluate the model
correct = 0  # Track number of correct predictions
total = 0  # Track total number of samples
with torch.no_grad():  # Disable gradient calculations (not needed for evaluation)
    for X_batch, y_batch in test_loader:
        outputs = model(X_batch)  # Get model predictions
        _, predicted = torch.max(outputs, 1)  # Get class with highest probability
        _, actual = torch.max(y_batch, 1)  # Get actual class labels
        correct += (predicted == actual).sum().item()  # Count correct predictions
        total += y_batch.size(0)  # Count total samples

test_accuracy = correct / total * 100  # Calculate accuracy percentage
print(f"Test Accuracy: {test_accuracy:.2f}% 🎉")
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e3f914-ea3c-8008-a0d2-464ef8e20ece)

3️⃣ **Click Run (▶) and check the output!** 

✅ Model evaluation complete! You should now see the test accuracy displayed as a percentage. 🏆🎉

---

## 🔮 8. Making Predictions

### **➕🐍 Add a New Code Cell**         

1️⃣ Click **+ Code** in the top left to add a new code cell.<br>
2️⃣ Copy and paste the following code into the new code cell.<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e3f98b-28d8-8002-8cd5-4291967bcf42)

```python
# Import NumPy for numerical computations
import numpy as np

# Example input features
new_sample = torch.tensor([[5.0, 3.2, 1.3, 0.2]], dtype=torch.float32)

# Get model prediction
prediction = model(new_sample)  

# Find class with highest probability
predicted_class = torch.argmax(prediction).item()  

# Print the predicted class
print(f"Predicted class: {predicted_class} 🎯")  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e3f98b-28d8-8002-8cd5-4291967bcf42)

3️⃣ **Click Run (▶) and check the output!** 

✅ Prediction complete! You should now see the predicted class for the new sample. 🎯🎉

---
## 🎯 9. Wrap-Up & Next Steps

🎉 Congratulations! You’ve just built and trained your first Deep Neural Network! 🚀<be>

✅ Loaded and prepared the dataset 📂<br>
✅ Built a deep learning model 🏗️<br>
✅ Trained the model 🔄<br>
✅ Evaluated its accuracy 📊<br>
✅ Made predictions 🎯<br>

📌 Next Workshop: [💬 Introduction to LLMs](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/personal-computer-intro-llms)

### 🔗 **Additional AI Resources** 📚   

- [Google Colab Guide](https://colab.research.google.com/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp)(Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>   
- [Google Machine Learning Crash Course - Neural Networks](https://developers.google.com/machine-learning/crash-course/neural-networks)   
- [IBM Deep Learning](https://www.ibm.com/think/topics/deep-learning)

🚀 **Keep learning and see you at the next workshop!** 🎉

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---