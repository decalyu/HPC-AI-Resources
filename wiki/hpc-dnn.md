# 🚀✨ HPC: Deep Neural Network (DNN) Workshop! 🌟✨

## 🎯 **Goal**
📊 Learn how to build, train, and visualize a **Deep Neural Network (DNN)** using **PyTorch and HPC**. We will use the **Places365 dataset** and leverage **multi-GPU distributed training** on **CSUSB's HPC cluster**. No prior experience is needed—just bring your curiosity and creativity! 🚀

---

## 📌 **What You Will Learn** 🧠💡  

✅ Understanding large datasets and their significance 📂 <br>
✅ Preparing and processing real-world data at scale 🔍 <br>
✅ Building a **Deep Neural Network** optimized for **HPC** 🏗️ <br>
✅ Training and evaluating a **DNN using multi-GPU acceleration** 🔄 <br> 
✅ Visualizing training results 📈  <br>
✅ Hands-on coding with **PyTorch, DistributedDataParallel (DDP), and HPC** 💻  <br>

---

## 🔑 **1. Key Terminologies in High-Performance Computing (HPC) Deep Neural Networks** 🧠  

🖥️ CPU vs. GPU

This image illustrates the difference between a CPU (Central Processing Unit) and a GPU (Graphics Processing Unit):

<img width="450" alt="Screenshot 2025-03-11 at 6 07 07 PM" src="https://github.com/user-attachments/assets/af1f80eb-a9e7-451a-bf22-22ad55372859" />


**CPU (left side)**: Contains only a few powerful cores optimized for sequential processing. It handles general-purpose computing tasks efficiently but struggles with massively parallel workloads.

**GPU (right side)**: Contains thousands of smaller cores designed for parallel processing. GPUs are highly efficient for matrix operations and deep learning computations, making them ideal for high-performance computing (HPC) tasks.

By leveraging GPUs over CPUs, deep learning models can process massive datasets faster and scale training workloads efficiently.

#### 🚀 **Parallel Computing**

- The technique of performing multiple computations simultaneously using multiple processors or cores.
- Essential for deep learning models, allowing faster training by distributing workloads across GPUs or nodes.

#### 🚀 **Distributed Computing**

- Splitting tasks across multiple nodes (computers) in an HPC cluster to enhance computational efficiency.
- Used in deep learning for training large models with massive datasets.

#### 🚀 **Tensor Cores**

- Hardware components within modern GPUs designed to accelerate matrix multiplications, crucial for AI workloads.
- Provide significant speedups in deep learning computations.

#### 🚀 **Why These Concepts Matter?**
- These HPC concepts **enable large-scale AI training**, reducing training times from weeks to hours.
- They allow deep learning models to **scale efficiently**, making cutting-edge AI research possible.

By leveraging **parallel processing, distributed computing, and GPU acceleration**, we can train deep learning models on massive datasets faster than ever before! 🚀  

---
## 🔍 **1: Access HPC Terminal via JupyterHub**

We will use CSUSB’s High-Performance Computing (HPC) system to run our AI code. Follow these steps to access JupyterHub on HPC.

## 🚀 **Step 1**: Log In to HPC with CI Logon 🔐

Let’s get you authenticated! Here’s how:

1️⃣ Go to [CSUSB HPC](https://csusb-hpc.nrp-nautilus.io/) if you're a student or teacher at CSUSB. If not, ask a teacher from your school to create an account for you using the [ACCESS CI](https://access-ci.org/get-started/for-educators/) program, which provides free access to computing tools like HPC Jupyter for classroom use.<br>
2️⃣ Click CI Logon to log in using your school account.<br>
3️⃣ Select GPU & Image.<br>

Once logged in:
- Under **GPU Type**, select **RTX A5000**.
- Choose **2 GPU cores**.
- In the **Image selection**, pick **Stack PyTorch 2**.

4️⃣ After logging in, Welcome to JupyterLab.

---

## 🔍 **2. Hands-on: Loading a Large Dataset on HPC**

Once we have access to JupyterHub we can begin loading our dataset!

## 🖥️ **Step 2**: Launch Your JupyterHub Server

Your HPC Jupyter environment is ready—let’s start coding!

1️⃣ Check Out the Launcher Page

- You’ll see several options like:
- Notebook: Start a Jupyter Notebook (e.g., Python 3 🐍).
- Console: Open a Python console for quick commands 📟.
- Other: Access Terminal, Text File, Markdown File, or Help 📚.

2️⃣ Open a Notebook or File

- Click Notebook → Python 3 (ipykernel) to start coding! ✍️
- You can also browse and open existing files. 📂

3️⃣ Run Your Code & Save Your Work

- Type your Python code and press Shift + Enter to run.
- Save often to keep your work safe! 💾

❓ **Why Use 2 GPU Cores and RTX A5000?**

We are working with **huge datasets** that require powerful GPUs for efficient computation. 

## 📂 Step 3: Load the Places365 Dataset

🏞️ **Places365** (PyTorch)
- A large-scale dataset used for **scene classification**.
- Contains **~1.8 million** images spanning **365 categories**.
- Needs **significant GPU power** for training deep neural networks.

### **➕🐍 Add a New Code Cell**    

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d31c30-d214-8008-a4de-4b2ba8a25cea)

```python
import torch  # Import PyTorch library
import torchvision  # Import torchvision for dataset handling
import torchvision.transforms as transforms  # Import transforms for preprocessing

# Check GPU availability and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to 256x256
    transforms.ToTensor(),  # Convert images to tensor format
    transforms.Normalize((0.5,), (0.5,))  # Normalize images to range [-1, 1]
])

# Load the Places365 training dataset with transformations applied
train_dataset = torchvision.datasets.Places365(
    root="/scratch/datasets",  # Path to dataset storage
    split='train-standard',  # Use standard training set
    download=True,  # Download dataset if not already present
    transform=transform  # Apply defined transformations
)

# Load the Places365 validation dataset with the same transformations
test_dataset = torchvision.datasets.Places365(
    root="/scratch/datasets",  # Path to dataset storage
    split='val',  # Use validation set
    download=True,  # Download dataset if not already present
    transform=transform  # Apply defined transformations
)

# Create DataLoader for efficient data processing in training
train_loader = torch.utils.data.DataLoader(
    train_dataset,  # Use training dataset
    batch_size=512,  # Process images in batches of 512
    shuffle=True,  # Shuffle data to improve model learning
    num_workers=16,  # Use 16 worker threads for data loading
    pin_memory=True  # Pin memory to optimize data transfer to GPU
)

# Create DataLoader for validation dataset
test_loader = torch.utils.data.DataLoader(
    test_dataset,  # Use validation dataset
    batch_size=512,  # Process images in batches of 512
    shuffle=False,  # No need to shuffle validation set
    num_workers=16,  # Use 16 worker threads for data loading
    pin_memory=True  # Pin memory to optimize data transfer to GPU
)

# Print dataset information
print(f"Dataset Loaded: Places365 with {len(train_dataset)} training images")

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d31c30-d214-8008-a4de-4b2ba8a25cea)

3️⃣ **Click Run (▶) and check the output!** 

✅ Dataset Loaded Successfully! You should now see the number of training images available in the Places365 dataset. 🏞️📊🎉

---

## 🏗️ **3. Building and Training a Deep Neural Network on HPC**

### Define a CNN Model

### **➕🐍 Add a New Code Cell**    

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d31c73-0a78-8008-9741-6d3899048a62)

```python
import torch.nn as nn  # Import neural network module from PyTorch
import torch.optim as optim  # Import optimizers for training

# Define a deep convolutional neural network (CNN)
class DeepCNN(nn.Module):
    def __init__(self):
        super(DeepCNN, self).__init__()  # Initialize parent class (nn.Module)

        # First convolutional layer: input channels = 3 (RGB), output channels = 64
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()  # ReLU activation function

        # Second convolutional layer: input channels = 64, output channels = 128
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        
        # Max pooling layer: downsample feature maps by a factor of 2
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layer 1: flatten input size = (128 * 128 * 128), output size = 1024
        self.fc1 = nn.Linear(in_features=128 * 128 * 128, out_features=1024)

        # Fully connected layer 2 (output layer): output size = 365 (Places365 categories)
        self.fc2 = nn.Linear(in_features=1024, out_features=365)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))  # Apply Conv1 -> ReLU -> MaxPool
        x = self.pool(self.relu(self.conv2(x)))  # Apply Conv2 -> ReLU -> MaxPool
        x = x.view(-1, 128 * 128 * 128)  # Flatten tensor for fully connected layers
        x = self.relu(self.fc1(x))  # Fully connected layer with ReLU activation
        x = self.fc2(x)  # Output layer (logits)
        return x

# Create an instance of DeepCNN and move it to the available device (CPU or GPU)
model = DeepCNN().to(device)
print("Model created!")

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d31c73-0a78-8008-9741-6d3899048a62)

3️⃣ **Click Run (▶) and check the output!** 

✅ Model created successfully! Your deep CNN is now ready for training. 🧠📊🎉

---

## 🏗️ **4. Evaluating the Model**

### Evaluate Model on Test Data

### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:   

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d31c9f-9fb8-8008-9b5d-6ed60e3c83c4)

```python
# Initialize counters for correct predictions and total samples
correct = 0
total = 0

# Disable gradient calculation to save memory and speed up inference
with torch.no_grad():
    for inputs, labels in test_loader:  # Iterate through test dataset
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU (if available)
        
        outputs = model(inputs)  # Forward pass: get model predictions
        
        _, predicted = torch.max(outputs, 1)  # Get class index with highest score
        
        total += labels.size(0)  # Count total samples
        correct += (predicted == labels).sum().item()  # Count correct predictions

# Calculate test accuracy as a percentage
accuracy = 100 * correct / total
print(f"Test Accuracy: {accuracy:.2f}% 🎯")

```
🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d31c9f-9fb8-8008-9b5d-6ed60e3c83c4)

3️⃣ **Click Run (▶) and check the output!** 

✅ Model evaluation complete! You should now see the test accuracy displayed as a percentage. 🎯📊🎉

---

## 🔮 5. Making Predictions

### Making Predictions

### **➕🐍 Add a New Code Cell**    

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:   

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d31cbd-fa98-8008-8961-ca1d34ad06d3)

```python
import numpy as np  # Import NumPy for numerical operations

# Generate a random test image with shape (1, 3, 256, 256) to simulate an input sample
new_sample = torch.randn(1, 3, 256, 256).to(device)  # Create a random tensor and move to device

# Perform forward pass through the model to get predictions
prediction = model(new_sample)

# Get the index of the highest probability class
predicted_class = torch.argmax(prediction, dim=1).item()

# Print the predicted class index
print(f"Predicted Class: {predicted_class} 🎯")

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d31cbd-fa98-8008-8961-ca1d34ad06d3)


3️⃣ **Click Run (▶) and check the output!** 

✅ Prediction complete! You should now see the predicted class for the random test image. 🎯📊🎉

---

## 🎉 6. Wrap-Up & Next Steps

🎯 Congratulations! You’ve just built and trained your first Deep Neural Network on HPC! 🚀

✅ Loaded and prepared the dataset 📂<br>
✅ Built a deep learning model optimized for multi-GPU HPC training 🏗️<br>
✅ Trained the model using distributed computing 🔄<br>
✅ Evaluated its accuracy on Places365 📊<br>
✅ Made predictions using the trained model 🎯<br>

📌 **Next Workshop** : [🚀 Intro to HPC, AI & Jupyter](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/hpc-intro-llms)

### **🔗 Additional AI Resources** 📚

- [Project Jupyter Documentation](https://docs.jupyter.org/en/latest/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>      
- [Fundamentals of Facial Recognition](https://learn.microsoft.com/en-us/training/modules/detect-analyze-faces/)
- [ACCESS CI](https://access-ci.org/get-started/for-educators/) (Free access to HPC for all using the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) U.S. government program)

🚀 Keep learning and see you at the next workshop! 🎉
---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---
