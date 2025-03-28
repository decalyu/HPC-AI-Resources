# 🚀✨ **HPC: 🤖 Simple ML Model!** ✨🚀  

## 🎯 **Goal**  
🎉 Welcome, future AI builder! In this hands-on workshop, you'll learn how to **build, train, and use a simple Machine Learning (ML) model** using Python on an **HPC (High-Performance Computing) system**. No prior experience is needed—just curiosity and excitement! 🤖📊  

---

## 📌 **What You Will Learn** 🧠💡  
✅ What is a **ML** model 🏗️  
✅ Why **HPC** for ML training ⚡  
✅ What **Linear Regression** is and how it works 📈  
✅ How to import **data** for ML 📥  
✅ How to **train** a simple ML model using HPC 🤖  
✅ How to make **predictions** with the model 🔮  
✅ How to **visualize** ML predictions 🎨📊  

---

## 🤖 **1. What is a ML Model?**  
### **ML = Computers that Learn!** 💡  
A **ML model** is a program that **learns from data** to make predictions. Instead of being manually programmed with rules, ML models **find patterns** in data and use them to make future decisions.  

💡 **Where is ML used?**  
- 🛍️ **Amazon & Shopify** → Suggests products based on your shopping history  
- 📺 **Netflix & YouTube** → Recommends movies & videos you might like  
- 🦱 **Face Unlock** → Recognizes your face using AI  

💡 **Quick Thought:** Can you think of other places you see ML in action? 🤔  

---

## ⚡ **2. Why Use HPC for ML?**  
High-Performance Computing (HPC) is like a **supercomputer** that can do many calculations **very fast**. ML models need a lot of calculations, and normal computers can take a long time to process them. With HPC, we can:   

✅ Train ML models **faster** ⏩  
✅ Work with **bigger datasets** 📊  
✅ Run **complex AI programs** 🤖  

### 💡 **Think of HPC Like a Super-Powered Kitchen**  
Imagine making **100 pizzas** by yourself. It would take **forever**! 🍕😫  
Now imagine you have **100 chefs** working together—that's **HPC**! They divide the work, cook faster, and get the job done efficiently. 🔥👨‍🍳  

In the same way, HPC **splits big ML tasks** across many powerful computers, making them run **much faster**! 🚀  

---

## 📈 **3. What is Linear Regression?**  
Linear Regression helps us **find patterns** in data and **make predictions** by drawing a straight line through the data points. 📊  

🔍 **Example:** Imagine tracking house features and their prices:  
- A house with **3 rooms** costs **$200,000**  
- A house with **4 rooms** costs **$250,000**  
- A house with **5 rooms** costs **$300,000**  

📌 **Pattern:** More rooms generally mean higher prices.  

📌 **Linear Regression** finds the best rule (line) to predict new prices.  

💡 **That’s Linear Regression!** 🎉  

---

## 🔍 **4. Hands-on: Preparing a Dataset for ML Model on a HPC**  

### 🚀 **Step 1: Access the HPC System**  
1️⃣ Go to [CSUSB HPC](https://csusb-hpc.nrp-nautilus.io/) if you're a student or teacher at CSUSB. If not, ask a teacher from your school to create an account for you using the [ACCESS CI](https://allocations.access-ci.org/get-your-first-project) program, which provides free access to computing tools like HPC Jupyter for classroom use.    
2️⃣ Log in.  
3️⃣ Increase the number of GPUs to 2.  
4️⃣ Select Stack PySpark for the image.  
5️⃣ Launch a Jupyter Notebook instance.  

🔹 **Why use Jupyter Notebook on HPC?**  
- It lets us run Python code **easily**.  
- It helps us **organize** our work.  
- It connects directly to the **HPC system** to run models faster.  

### 💾 **Step 2: Load and Explore the Dataset**  

### **➕🐍 Add a New Code Cell**        

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:   

**🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d10c58-efc8-8009-b304-71747f899693)**

```python
# Pandas helps us work with data in tables (like spreadsheets).
import pandas as pd  
# This gives us the California Housing dataset to train our model.
from sklearn.datasets import fetch_california_housing  

# Step 1: Load the California Housing dataset from sklearn
california_housing = fetch_california_housing()

# We get the features (X) and target (y) values from the dataset
# Features (such as AveRooms, AveOccup, etc.)
X = california_housing.data  
# Target variable (house values)
y = california_housing.target  

# Step 2: Convert the features and target to a pandas DataFrame for easier handling
df = pd.DataFrame(X, columns=california_housing.feature_names)

# Add the target values (house prices) to the DataFrame
df['House Value'] = y

# Step 3: Display the first few rows of the cleaned data to check our work
print(df.head())

# Step 4: Choose 'AveRooms' as the feature (X) and 'House Value' as the target (y)
# Use AveRooms to predict house prices
X = df[['AveRooms']].values  
# House prices we want to predict
y = df['House Value'].values  

```  

**🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d10c58-efc8-8009-b304-71747f899693)**

3️⃣ **Click Run (▶) and check the output!**

✅ California Housing dataset loaded successfully! You should now see a preview of the cleaned data. 🏡📊🎉
 
---

## 🤖 **5. Training a Simple ML Model on HPC**  

###  **Train a Simple ML Model using Linear Regression**  

### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

**🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d10d4a-6810-8009-bda6-f9298d3d05a7)**

```python
#Install packages not included in the stack   
!pip install --no-cache-dir torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118   
!pip install --upgrade typing_extensions   

# PyTorch helps us create and train ML models.
import torch  
# This part is specifically for building neural networks (models like linear regression).
import torch.nn as nn  
# This part helps adjust the model during training to make it more accurate.
import torch.optim as optim  
# This helps us split our data into a training set and a testing set.
from sklearn.model_selection import train_test_split  
# This helps to adjust the data to make it easier for the model to understand.
from sklearn.preprocessing import StandardScaler  

# Step 5: Split the data into training and testing sets (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Standardize the data so that all features have a mean of 0 and a standard deviation of 1
# Create a scaler object
scaler = StandardScaler()  
# Fit and transform training data
X_train_scaled = scaler.fit_transform(X_train)  
# Use the same transformation for test data
X_test_scaled = scaler.transform(X_test)       

# Step 7: Convert the scaled data into PyTorch tensors (which is the format the model uses)
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
# Convert targets to column vectors
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1) 
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)    

# Step 8: Define the linear regression model (a simple neural network with 1 layer)
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        # Define the linear layer
        self.linear = nn.Linear(input_dim, 1)  

    def forward(self, x):
        # Return the model's output
        return self.linear(x)  

# Step 9: Instantiate the model (create an object of the LinearRegressionModel class)
# input_dim is the number of features in X
model = LinearRegressionModel(X_train.shape[1])  

# Step 10: Define the loss function (how we measure the error) and optimizer (how we adjust the model)
# Mean Squared Error loss function
criterion = nn.MSELoss()  
# Stochastic Gradient Descent optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)  

# Step 11: Train the model
# Number of times we will loop through the training data
num_epochs = 1000  
for epoch in range(num_epochs):
    # Forward pass: Compute predicted y by passing X through the model
    y_pred = model(X_train_tensor)

    # Calculate the loss (how far off our predictions are)
    loss = criterion(y_pred, y_train_tensor)

    # Backward pass: Calculate gradients
    # Clear previous gradients
    optimizer.zero_grad()  
    # Backpropagate the gradients
    loss.backward()        

    # Update the model's weights based on the gradients
    optimizer.step()

    # Print loss every 100 epochs to see the progress
    if (epoch+1) % 100 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Step 12: Evaluate the model on the test set (check how well it generalizes)
with torch.no_grad():
    # Get predictions for the test data
    y_pred_test = model(X_test_tensor)  
    # Calculate the loss on the test set
    test_loss = criterion(y_pred_test, y_test_tensor)  
    print(f'Test Loss: {test_loss.item():.4f}')

```  

**🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d10d4a-6810-8009-bda6-f9298d3d05a7)**

3️⃣ Click **Run** (▶) to train the ML model!  

✅ Model training complete! You should now see the loss decreasing over epochs and the final test loss displayed. 🔥📉🎉

🔹 Why Use HPC Here?

- Training a model on a regular laptop could take hours! 🕒
- On HPC, it can take minutes! ⚡

---

## 🔮 **6. Making Predictions with HPC**  

### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

**🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d10e0f-f620-8009-b0dd-1b43e5d15bce)**

```python
# Step 13: Make predictions on the training set
with torch.no_grad():
    # Get predictions for the training data
    y_pred_train = model(X_train_tensor)  

# Step 14: Predict the price for a house with 15 rooms
# First, scale the input value (15 rooms) using the same scaler
scaled_15_rooms = scaler.transform([[15]])   

# Convert the scaled value to a PyTorch tensor and make the prediction
scaled_15_rooms_tensor = torch.tensor(scaled_15_rooms, dtype=torch.float32)
# Get the price prediction for 15 rooms
predicted_price = model(scaled_15_rooms_tensor).item()  

# Print the predicted price for a house with 15 rooms
print(f"The predicted price for a house with 15 rooms is: ${predicted_price* 100000:,.2f}")

```  

**🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d10e0f-f620-8009-b0dd-1b43e5d15bce)**

3️⃣ Click **Run** (▶) to see the prediction!  

✅ Prediction complete! You should now see the estimated house price for a home with 15 rooms. 🏡💰🎉

🎯 **Quick Challenge**
* Change the number of rooms for the prediction and see the results change!

---

## 🎨 **7. Visualizing ML Predictions**  
### **➕🐍 Add a New Code Cell**   
 
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

**🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67d10e6f-8fe4-8009-8115-e320b42510a0)**

```python
# NumPy helps with numerical operations, like handling arrays.
import numpy as np  

# This helps us create graphs to visualize the data and predictions.
import matplotlib.pyplot as plt  

# Step 15: Plot the results
# Set the size of the plot
plt.figure(figsize=(8, 6))  

# Plot the original data (scatter plot of AveRooms vs House Value)
plt.scatter(df['AveRooms'], df['House Value'], alpha=0.5, label="Data points")

# Step 16: Plot the regression line
# Sort the 'AveRooms' values so we can plot the regression line in a smooth manner
sorted_ave_rooms = np.sort(df['AveRooms'])

# Scale the sorted AveRooms values (so they match the model's input format)
scaled_ave_rooms = scaler.transform(sorted_ave_rooms.reshape(-1, 1))

# Get the predicted house values for the sorted, scaled 'AveRooms'
predicted_house_value = model(torch.tensor(scaled_ave_rooms, dtype=torch.float32)).detach().numpy()

# Plot the regression line (predicted values)
plt.plot(sorted_ave_rooms, predicted_house_value, color='red', label='Regression Line', linewidth=2)

# Step 17: Add labels and title to the plot
plt.title('AveRooms vs House Value with Linear Regression Line')
plt.xlabel('Average Rooms')
plt.ylabel('House Value (100,000)')
plt.legend()  # Show the legend

# Step 18: Display the plot
plt.show()

```  

**🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d10e6f-8fe4-8009-8115-e320b42510a0)**

3️⃣ Click **Run** (▶) and check the graph! 🎨📊  

✅ Visualization complete! You should now see a scatter plot with a regression line showing the relationship between average rooms and house value. 📊🎨🎉

---

## 🎯 **8. Wrap-Up & Next Steps**

🎉 Great job! You just built, trained, and tested your first ML model on an **HPC system**!  

✅ ML helps computers learn from data 🤖📊  
✅ Linear Regression finds patterns and predicts outcomes 📈  
✅ We trained an ML model to predict the value of homes based on the number of rooms 🏆  
✅ We visualized predictions with a graph 🎨📊  

🚀 Next Workshop: [🧠 Deep Neural Network (DNN)](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/hpc-dnn)   


### **🔗 Additional AI Resources** 📚

- [Project Jupyter Documentation](https://docs.jupyter.org/en/latest/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>      
- [Microsoft Learn - Introduction to Machine Learning](https://learn.microsoft.com/en-us/training/modules/introduction-to-machine-learning/)
- [ACCESS CI](https://access-ci.org/get-started/for-educators/) (Free access to HPC for all using the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) U.S. government program)

🎉 **You did it! Now you know how HPC makes ML training faster! Keep exploring AI, and see you at the next workshop!** 🚀

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---