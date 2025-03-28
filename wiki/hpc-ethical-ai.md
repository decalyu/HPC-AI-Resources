# 🚀✨ HPC: Ethical AI & Future Trends on HPC Workshop! 🌟✨

## 🎯 **Goal**

📊 Learn how to develop **Ethical AI** models while leveraging **High-Performance Computing (HPC)** for large-scale training. We will use **The LAION-5B dataset**, one of the largest open-source image-text datasets, requiring HPC resources for ethical AI model training and analysis. 🚀


## 📌 **What You Will Learn** 🧠💡  

✅ Understanding **Ethical AI** and its challenges (bias, fairness, explainability) 🤖 <br>
✅ Exploring **HPC** as a solution for training large AI models responsibly 💻 <br>
✅ Training AI models on large-scale datasets using **multi-GPU HPC clusters** 🏗️ <br> 
✅ Evaluating AI fairness, transparency, and accountability 🔍  <br>
✅ Future trends in Ethical AI and responsible AI development 🌍  <br>

---

## 📚 **1. Key Terminologies in Ethical AI & HPC** 🔑

### 🌍 **Ethical AI**
- AI that ensures **fairness, transparency, and accountability** in decision-making.
- Addresses bias in datasets and models to prevent discrimination.

### 🚀 **HPC for AI**
- **High-Performance Computing (HPC)** enables AI training on massive datasets by distributing computations across GPUs and multiple nodes.
- Essential for training **foundation models** and **large-scale NLP/CV applications**.

### 🏗️ **Bias & Fairness in AI**
- **Algorithmic Bias**: When AI models make unfair decisions based on skewed data.
- **Fairness Metrics**: Statistical Parity, Equal Opportunity, Equalized Odds, Disparate Impact.
- **Debiasing Techniques**: Data preprocessing, adversarial training, fairness constraints.

### 💻 **Federated Learning & Privacy-Preserving AI**
- **Federated Learning**: AI training across distributed devices while keeping data localized.
- **Differential Privacy**: Adding noise to AI models to protect sensitive data.

### 🌍 **Future Trends in Ethical AI**
- **Explainable AI (XAI)**: Making AI decision-making understandable.
- **Regulatory Frameworks**: EU AI Act, AI Ethics Guidelines from NIST, UNESCO.
- **Sustainable AI**: Reducing AI's carbon footprint through efficient HPC utilization.

---

## 🔍 **2: Access HPC Terminal via JupyterHub**

1️⃣ Go to [CSUSB HPC](https://csusb-hpc.nrp-nautilus.io/) if you are a learner or educator at CSUSB. Otherwise, have an educator from your school create an account for you using the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support [ACCESS CI](https://access-ci.org/get-started/for-educators/), a U.S. government program that provides free access to HPC resources.<br>
2️⃣ Click **CI Logon** to log in using your school account.<br>
3️⃣ Select the GPU model that best fits your needs.<br>
4️⃣ After logging in, Welcome to JupyterLab.<br>
✅ You're ready to go!

---

## 🔍 **3. Hands-on: Loading a Large Dataset on HPC**

We will use CSUSB's High-Performance Computing (HPC) system to run our AI code. Follow these steps to access JupyterHub on HPC.

## 🚀 **Step 1**: Log In to HPC with [CI Logon](https://csusb-hpc.nrp-nautilus.io/) 🔐

Let's get you authenticated! Here's how:

1️⃣ Go to the CI Logon Portal

- Open [CI Logon](https://csusb-hpc.nrp-nautilus.io/) in your browser.
- Click Sign In with CI Logon

2️⃣ Select Your Identity Provider & Log In

- Choose "California State University, San Bernardino" 🎓 from the dropdown.
- Check "Remember this selection" to save time next login. ✅
- Click Log in to proceed. 🚀

## 🖥️ **Step 2**: Launch Your JupyterHub Server

Your HPC Jupyter environment is ready—let's start coding!

1️⃣ Check Out the Launcher Page

- You'll see several options like:
- Notebook: Start a Jupyter Notebook (e.g., Python 3 🐍).
- Console: Open a Python console for quick commands 📟.
- Other: Access Terminal, Text File, Markdown File, or Help 📚.

2️⃣ Open a Notebook or File

- Click Notebook → Python 3 (ipykernel) to start coding! ✍️
- You can also browse and open existing files. 📂

3️⃣ Run Your Code & Save Your Work

- Type your Python code and press Shift + Enter to run.
- Save often to keep your work safe! 💾

❓ **Why Use 2 GPUs and RTX A5000?**

We are working with **huge datasets** that require powerful GPUs for efficient computation. 

## 📂 Step 3: Load **LAION-5B**, a Massive Multimodal Dataset

📸 **LAION-5B**

- A massive dataset with **5 billion** image-text pairs.
- Used for training **vision-language models** like CLIP.
- Demands **high memory and computational power** for data processing.

### **➕🐍 Add a New Code Cell**  
  
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4be16-9ac4-8011-8764-edc272c960e7)

```python
import torch  # Import PyTorch for tensor operations and model training
import torchvision  # Import torchvision for dataset handling
import torchvision.transforms as transforms  # Import transforms for image preprocessing

# Check if a GPU is available; otherwise, use the CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define dataset transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Resize images to a fixed size of 256x256 pixels
    transforms.ToTensor(),  # Convert images to PyTorch tensors (with values in [0,1])
    transforms.Normalize((0.5,), (0.5,))  # Normalize image pixel values to range [-1, 1]
])

# Load the LAION-5B dataset (assuming images are organized in folders by category)
train_dataset = torchvision.datasets.ImageFolder(
    root="/scratch/datasets/LAION-5B",  # Path to dataset directory
    transform=transform  # Apply transformations to each image
)

# Create a DataLoader to efficiently load and process images during training
train_loader = torch.utils.data.DataLoader(
    train_dataset,  # Use the loaded dataset
    batch_size=512,  # Load images in batches of 512 for efficient processing
    shuffle=True,  # Shuffle dataset order for better training performance
    num_workers=16,  # Use 16 worker threads for faster data loading
    pin_memory=True  # Optimize memory usage when using a GPU
)

# Error handling before printing dataset statistics
try:
    dataset_size = len(train_dataset)
    print(f"Dataset Loaded: LAION-5B with {dataset_size} images")
except Exception as e:
    print(f"Error loading dataset: {e}")
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e4bddd-c608-8011-8e0b-ec3d21bbcd7d)

3️⃣ **Click Run (▶) and check the output!** 

✅ Dataset Loaded Successfully! You should now see the number of images available in the LAION-5B dataset. 🖼️📊🎉

---

## 🏗️ **3. Building and Training a Deep Learning Model for Ethical AI**

## 🚀 Step 1: Define a Bias-Aware Vision Transformer (ViT) Model

### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4be72-52a8-8011-9d84-e0375582f41c)

```python
from transformers import ViTForImageClassification, ViTFeatureExtractor  # Import ViT model and feature extractor
import torch.nn as nn  # Import neural network module from PyTorch

# Load pre-trained Vision Transformer (ViT) model
model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

# Move the model to the available device (GPU if available, otherwise CPU)
model.to(device)

# Print confirmation message
print("Pre-trained Vision Transformer Model Loaded! 🚀")
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d320de-a97c-8008-8699-cb158b0c191c)

3️⃣ **Click Run (▶) and check the output!** 

✅ Pre-trained Vision Transformer Model Loaded! Your ViT model is now ready for image classification. 🚀🎉

## 🚀 Step 2: Train the Model with Ethical AI Constraints

### **➕🐍 Add a New Code Cell**
    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bedf-b9d4-8011-9376-e2ee4c12a225)

```python
import torch.optim as optim  # Import optimizers from PyTorch

# Define loss function (CrossEntropyLoss for classification tasks)
criterion = nn.CrossEntropyLoss()

# Define optimizer (Adam with a learning rate of 0.0001)
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop for 5 epochs
for epoch in range(5):
    running_loss = 0.0  # Initialize cumulative loss for the epoch
    
    for inputs, labels in train_loader:  # Iterate over batches in the training set
        inputs, labels = inputs.to(device), labels.to(device)  # Move data to GPU (if available)
        
        optimizer.zero_grad()  # Reset gradients before each batch
        outputs = model(inputs).logits  # Forward pass (ViT's output is stored in `.logits`)
        loss = criterion(outputs, labels)  # Compute loss
        loss.backward()  # Backpropagate gradients
        optimizer.step()  # Update model parameters
        
        running_loss += loss.item()  # Accumulate loss
        
    # Print loss at the end of each epoch
    print(f"Epoch {epoch+1}: Loss {running_loss:.4f}")

print("Training Complete! 🚀")
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d3210a-0f70-8008-9557-cccb6f04276a)

3️⃣ **Click Run (▶) and check the output!** 

✅ Training Complete! Your Vision Transformer model has been trained for 5 epochs. 🚀📊🎉

🎯 Challenge: What if you train longer—or too long? Try doubling the number of epochs or increasing the batch size. Does your model actually get better… or worse? 🤔📉
💡 Extra Tip: Edit `range(5)` in the training loop and `batch_size=512` in the DataLoader.

---

## 🏆 **4. Evaluating Model Fairness & Bias Mitigation**

### Evaluate Model Performance Across Demographics

### **➕🐍 Add a New Code Cell**    

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bfbc-ad48-8011-aeb0-c6af61d99c32)

```python
import numpy as np  # Import NumPy for array operations
from sklearn.metrics import accuracy_score, confusion_matrix  # Import accuracy metric and confusion matrix

# Define a placeholder function for fairness evaluation
def evaluate_fairness(predictions, labels, sensitive_attribute):
    """
    Evaluate fairness by computing accuracy per sensitive attribute group.

    Args:
    - predictions (np.array): Model-predicted class labels.
    - labels (np.array): Ground-truth class labels.
    - sensitive_attribute (np.array): Array of sensitive attributes (e.g., gender, race).

    Returns:
    - dict: Accuracy per unique group in the sensitive attribute.
    """
    unique_groups = np.unique(sensitive_attribute)  # Get unique sensitive groups
    group_accuracies = {}  # Dictionary to store accuracy per group

    for group in unique_groups:
        group_indices = (sensitive_attribute == group)  # Get indices for the current group
        group_accuracy = accuracy_score(labels[group_indices], predictions[group_indices])  # Compute accuracy
        group_accuracies[group] = group_accuracy  # Store result

    return group_accuracies  # Return dictionary with fairness metrics

# Example usage (mock data for demonstration)
y_pred = np.random.randint(0, 10, size=len(train_dataset))  # Generate random mock predictions
y_true = np.random.randint(0, 10, size=len(train_dataset))  # Generate random mock ground truth labels
sensitive_attr = np.random.choice(["Male", "Female", "Other"], size=len(train_dataset))  # Assign random demographic groups

# Evaluate fairness across different groups
fairness_results = evaluate_fairness(y_pred, y_true, sensitive_attr)

# Print fairness evaluation results
print("Fairness Evaluation Results:", fairness_results)
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d3213c-6f40-8008-8452-6177d208a5a4)


3️⃣ **Click Run (▶) and check the output!** 

✅ Fairness evaluation complete! You should now see accuracy metrics per sensitive attribute group. 📊⚖️🎉

---

## 🎉 5. **Wrap-Up & Next Steps**

🎯 Congratulations! You’ve just built and trained an **Ethical AI Model** using **HPC**! 🚀<br>

✅ Loaded a **large-scale dataset (LAION-5B)** 📂 <br>
✅ Built a **bias-aware Vision Transformer model** 🏗️ <br>
✅ Trained the model using **HPC with multi-GPU acceleration** 🔄 <br>
✅ Evaluated **bias and fairness** across demographic groups 📊  <br>

### **🔗 Additional AI Resources** 📚

- [Project Jupyter Documentation](https://docs.jupyter.org/en/latest/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>      
- [Responsible AI by Microsoft](https://learn.microsoft.com/en-us/training/modules/embrace-responsible-ai-principles-practices/)
- [ACCESS CI](https://access-ci.org/get-started/for-educators/) (Free access to HPC for all using the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) U.S. government program)

🚀 Keep learning and see you at the next workshop! 🎉

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---