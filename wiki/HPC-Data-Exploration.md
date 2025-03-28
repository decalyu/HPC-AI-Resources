# 🚀✨ **HPC: 📊 Visualize AI Data!** ✨🚀

## 🎯 **Goal**

🚀 Learn to clean, analyze, and visualize real-world datasets on HPC using Python tools like Pandas, Seaborn, and Matplotlib. Gain hands-on experience in data preprocessing, chart creation, and result interpretation while working in JupyterLab. No prior experience needed—just bring your curiosity! 🌟

---
## 📌 **What You Will Learn 🧠💡**

✅ Access HPC Terminal via JupyterHub – Log in and navigate CSUSB HPC 🖥️<br>
✅ Install Dependencies and Download Dataset – Set up tools and fetch real-world data 📂🔍<br>
✅ Load and Clean the Dataset – Handle missing values and prepare data for analysis 🧹✨<br>
✅ Visualize Data with Charts – Create bar charts, scatter plots, and histograms 📊🎨<br>
✅ Save and Transfer Cleaned Data – Store processed data and download it locally 💾🔄<br>

---


## 🔍 **1: Access HPC Terminal via JupyterHub**

1️⃣ Go to [CSUSB HPC](https://csusb-hpc.nrp-nautilus.io/) if you're a student or teacher at CSUSB. If not, ask a teacher from your school to create an account for you using the [ACCESS CI](https://access-ci.org/get-started/for-educators/) program, which provides free access to computing tools like Jupyter for classroom use.<br>
2️⃣ Click **CI Logon** to log in using your school account.<br>
3️⃣ Select the GPU model that best fits your needs.<br>
4️⃣ After logging in, Welcome to JupyterLab.<br>
✅ You're ready to go!

---

## 💻 **2. Install Dependencies, Setup Kaggle, and Download Dataset**

### 📝 **Open Terminal in JupyterLab**
1️⃣ **Click Terminal to open a command-line interface** <br> 
2️⃣ In the terminal, paste and run the following commands:<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c81e-b498-8011-ac57-860a363fd761)

```bash
# Install essential Python libraries for data analysis and visualization
pip install --user pandas seaborn matplotlib 

# Create a dataset storage directory
mkdir -p ~/playstore_data && cd ~/playstore_data

# Download Kaggle dataset
curl -L -o google-playstore-apps.zip \
  https://www.kaggle.com/api/v1/datasets/download/gauthamp10/google-playstore-apps

# Unzip dataset
unzip google-playstore-apps.zip

# List extracted files
ls -lh
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11672-d9f8-8011-a6be-87ad2ad65d11)

3️⃣ **Click Run (▶) and check the output.**

✅ Dependencies installed, Kaggle configured, and dataset downloaded successfully! 🎉



---


## 💾 **3: Load the Dataset into Jupyter Notebook**

### **📂 Step 1: Open the /playstore_data/ Directory**
1️⃣ Navigate to **/playstore_data/** in JupyterLab.<br>
2️⃣ Inside the folder, create a new **Python 3 (ipykernel)** notebook.  

### **📂 Step 2: Run the Following Code in a New Cell**
1️⃣ In the next available cell, add the following code.<br>
2️⃣ Paste and run the following code:<br>

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c85a-b56c-8011-a5a8-8ce235f1dfbd)

```python
import pandas as pd  # Pandas helps us handle structured data

# Load the Google Playstore dataset into a DataFrame
playstore_df = pd.read_csv("~/playstore_data/Google-Playstore.csv")

# Display the first few rows of the dataset
playstore_df.head()
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d116b9-5134-8011-aad6-e89a0e557ef9)

3️⃣ **Click Run (▶) and check the output.**

✅ You should now see your dataset displayed! 🎉



---

## 🧹 **4. Cleaning the Data**

### **➕🐍 Add a New Code Cell**     
                         
1️⃣ Click **+ Code** in the top left to add a new code cell.        
2️⃣ Copy and paste the following into the new code cell.   

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c8a4-6af0-8011-b4cb-12d5df3f7bd5)

```python
# Check for missing values
print(playstore_df.isnull().sum())  # Shows count of missing values per column

# Remove missing values and duplicates
playstore_df.dropna(inplace=True)  # Remove rows with missing values
playstore_df.drop_duplicates(inplace=True)  # Remove duplicate rows

# Display cleaned dataset
playstore_df.head()  # Show cleaned data table

# Check all columns
print(playstore_df.columns) 
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11701-ff08-8011-8fc1-687e777accba)

3️⃣ **Click Run (▶) and check the output.**

✅ No more missing values! Everything is clean and ready for analysis.🎉

---

## 🎨  **5: Data Visualization**

### 📊 **Bar Chart: Trending Video Categories**

### **Step 1: ➕🐍 Add a New Code Cell**

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c8f4-47b4-8011-b2ca-9b14875071bf)

```python
import matplotlib.pyplot as plt  # Import Matplotlib for creating plots
import seaborn as sns  # Import Seaborn for advanced data visualization

# Set the figure size for better readability
plt.figure(figsize=(12, 8))
# Create a count plot to visualize the number of apps in each category
# `order` ensures that categories are displayed in descending order based on frequency
sns.countplot(y=playstore_df["Category"], order=playstore_df["Category"].value_counts().index)
# Set the title of the plot to describe the visualization
plt.title("Most Common App Categories")  
# Label the x-axis as "Count" to indicate the number of apps in each category
plt.xlabel("Count")  
# Label the y-axis as "Category" to show different app categories
plt.ylabel("Category")  
# Display the plot
plt.show()

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11733-c80c-8011-9c74-130a75d07642)

```

3️⃣ **Click Run (▶) and check the output.**

✅ This will show the distribution of trending videos across different categories🎉

<img src="https://github.com/user-attachments/assets/f1b02bca-fee0-474e-8e44-d92fe9586a74" width="500">

🎯 Challenge: Modify the dataset by changing the x-axis column in the bar chart from "Category" to "Content Rating".

💡 Extra Tip: Edit the `sns.countplot y=playstore_df["Category"]`.



### 📊 Scatter Plot: App Ratings vs. Number of Ratings**

### **Step 2: ➕🐍 Add a New Code Cell**

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c937-5bb0-8011-a599-9ffb8d911bfc)

```python
# Set the figure size to improve readability
plt.figure(figsize=(10, 5))
# Create a scatter plot showing the relationship between the number of ratings and the app rating
sns.scatterplot(x=playstore_df["Rating Count"], y=playstore_df["Rating"])
# Set the x-axis label to indicate the number of ratings an app has received
plt.xlabel("Number of Ratings")  
# Set the y-axis label to indicate the app's rating score
plt.ylabel("App Rating")  
# Set the title to provide context about the visualization
plt.title("Ratings vs. Rating Count")  
# Display the plot
plt.show()

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11768-5ba8-8011-88dc-47e2fd105b26)

3️⃣ **Click Run (▶) and check the output.**

✅ A dynamic scatter plot appears, revealing app rating trends! 🎉

<img src="https://github.com/user-attachments/assets/2d168f74-9db8-4acd-b07b-594f16a784ac" width="500">

### 📊 **Histogram: App Ratings**

### **Step 3: ➕🐍 Add a New Code Cell***

1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c978-46a0-8011-81de-d5097f57886d)

```python
# Create a histogram of app ratings
plt.figure(figsize=(12, 6))  # Set plot size
sns.histplot(playstore_df["Rating"], bins=30, kde=True)  # Histogram with distribution line
plt.title("Distribution of App Ratings")  # Chart title
plt.xlabel("Rating")  # Label for X-axis
plt.ylabel("Frequency")  # Label for Y-axis
plt.show()  # Display the plot
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d1179a-73b8-8011-918d-11a79164a854)

3️⃣ **Click Run (▶) and check the output.**

✅ A colorful histogram appears, showing app rating trends! 🎉

<img src="https://github.com/user-attachments/assets/49801895-3673-4c37-80f7-5a11123f975b" width="500">

---

## 💾 **6: Save and Transfer Cleaned Data**

### **Step 1: Save the Cleaned Dataset on HPC**

### **➕🐍 Add a New Code Cell**          
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c9c6-0c8c-8011-9401-017423ea205f)

```python
playstore_df.to_csv("~/playstore_data/cleaned_playstore_data.csv", index=False)

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d117e7-ddf8-8011-a918-334fe2d9dde7)

3️⃣ **Click Run (▶) and check the output.**


✅ The cleaned dataset is now saved on HPC🎉

### **Step 2: Download Data to Your Local Machine via JupyterLab**

1️⃣ Open JupyterLab in your browser.<br>
2️⃣ Navigate to the folder **`playstore_data`** where your cleaned dataset is stored.<br>
3️⃣ Right-click on **`cleaned_playstore_data.csv`** and select **Download**.<br>
4️⃣ The file will be saved to your local computer.<br>
✅ Now you have the cleaned dataset on your local machine! 🎉<br>

---

## 🎯 **7. Wrap-Up & Next Steps**

🎉 Congratulations! You learned how to:

✅ Access CSUSB HPC & JupyterLab 🖥️  
✅ Download and load real-world datasets 🔍📂  
✅ Clean missing values and preprocess data 🧹✨  
✅ Analyze and visualize data with Python 📊🎨  
✅ Save and transfer your work from HPC 🚀  

🚀 **Next Workshop: [🤖 Simple ML Model](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/hpc-simple-ml)**  

### 🔗 Additional AI & HPC Resources 📚   

- [Project Jupyter Documentation](https://docs.jupyter.org/en/latest/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>      
- [Get started with Microsoft data analytics](https://learn.microsoft.com/en-us/training/paths/data-analytics-microsoft/) 
- [ACCESS CI](https://access-ci.org/get-started/for-educators/) (Free access to HPC for all using the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) U.S. government program)

🎉 **You did it! Keep exploring AI, and see you at the next workshop!** 🚀

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---



