# 🚀✨ HPC: Introduction to Large Language Models (LLMs) ✨🚀

## 🎯 Goal
🤖 Learn how to efficiently run Large Language Models (LLMs) on **High-Performance Computing (HPC) systems**, leveraging **JupyterLab** for scalable text generation and question-answering tasks. Whether you're new to LLMs or HPC, this hands-on workshop will guide you through the essentials! 🚀

---

## 📌 What You Will Learn 🧠💡
✅ Why use **HPC** for running LLMs? 🚀  
✅ How to access and navigate **CSUSB HPC & JupyterLab** 🖥️  
✅ Install essential Libraries and Kaggle dataset ✍️  
✅ How to run **pre-trained LLMs on HPC** for faster processing ⏩  
✅ Using **Hugging Face Transformers** for text generation and QA  
✅ Working with a **Kaggle dataset** for real-world applications 📊  

---

## 🏆 1. Why use **HPC** for running LLMs? 🚀
### 🚀 The Power of High-Performance Computing (HPC)
LLMs are computationally expensive! Running them on personal devices can be **slow**, **memory-intensive**, and sometimes **impossible**. HPC systems solve this by:
- **Parallelizing workloads** for faster execution ⚡
- **Providing GPU acceleration** to handle large-scale deep learning models 🎮
- **Handling large datasets** efficiently 📊

💡 *Think: What AI tasks can benefit from HPC? Jot down your ideas! 📝*

---

## 🔥 2. How to access and navigate **CSUSB HPC & JupyterLab** 🖥️

Once you sign in to the CSUSB HPC portal, follow these steps to configure and launch your server:

### Step 1: Access the HPC JupyterHub   
1️⃣ Go to [CSUSB HPC](https://csusb-hpc.nrp-nautilus.io/) if you're a student or teacher at CSUSB. If not, ask a teacher from your school to create an account for you using the [ACCESS CI](https://access-ci.org/get-started/for-educators/) program, which provides free access to computing tools like Jupyter for classroom use.<br>
2️⃣ Click CI Logon and authenticate.

### Step 2: Configure Your Server   
1️⃣ Click Start My Server or Launch Server if prompted.   
2️⃣ Under Advanced Options, adjust the following:   

- GPUs: 2
- GPU Type: Leave as Any
- Cores: 4 (default)
- RAM: 16 GB (default)
 
3️⃣ Under Image, select:   
✅ Stack Datascience   
### Step 3: Start Your Server   
1️⃣ Scroll down and click Start to launch the server.   
2️⃣ Wait for the server to initialize. Once it is ready, JupyterHub will open in a new tab.

✅ Now your server is ready for the workshop! 🚀

---

## 🔧  3. Install essential Libraries and Kaggle dataset ✍   

### **➕ 🐍  Add a New Code Cell**  

1️⃣  Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣  Copy and paste the following code:  

🔗  [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e5022a-b808-800b-b70c-794ee6bc8662)

```bash

# Step 1: Install necessary Python libraries for data manipulation and analysis
# Install pandas, numpy, and other essential libraries
!pip install pandas numpy scipy scikit-learn

# Step 2: Install seaborn and matplotlib for data visualization
!pip install seaborn matplotlib

# Step 3: Install Kaggle API library to interact with Kaggle
!pip install kaggle

# Step 4: Set up the Kaggle API authentication
# Make sure to upload your kaggle.json API key to the appropriate directory.
# If you don't have a kaggle.json file, you can get it by going to:
# https://www.kaggle.com/docs/api and generating a new API token.

# You need to add your Kaggle API credentials (kaggle.json) to the system path
import os
# Place your kaggle.json in the location '/root/.kaggle/kaggle.json' (for Jupyter Notebook) or your user's Kaggle directory
os.environ['KAGGLE_CONFIG_DIR'] = '/root/.kaggle'  # Adjust path if running locally

# Step 5: Download the Google Playstore dataset from Kaggle using the Kaggle API
# Set the dataset name for Google Playstore dataset
dataset_name = 'google-playstore-dataset'
# Use Kaggle API to download the dataset
!kaggle datasets download -d 'google-playstore/google-playstore-dataset'

# Step 6: Unzip the downloaded dataset into a new directory
import zipfile

# Define the path where the dataset will be extracted
output_dir = 'google_playstore_data'
# Create the directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Unzip the downloaded dataset
with zipfile.ZipFile(f'{dataset_name}.zip', 'r') as zip_ref:
    zip_ref.extractall(output_dir)

# Step 7: Verify the contents of the extracted dataset
# Check the files that were extracted
extracted_files = os.listdir(output_dir)
print(f'Files extracted: {extracted_files}')

# Now the Google Playstore dataset is ready for use.

```

🔗  [ChatGPT explanation for the code](https://chatgpt.com/share/67e50288-c82c-800b-9fb7-808e9277bfb3)

3️⃣  **Click Run (▶) and check the output!** 

✅  **Success!** Dataset downloaded and ready for analysis. 🎉 

---

## 🤖  4. How to run **pre-trained LLMs on HPC** for faster processing ⏩  

## 🏗  Step 1: Load Dataset into Jupyter Notebook

1️⃣  In **JupyterLab**, open a **new notebook** (Python 3 Kernel).  
2️⃣  In the first code cell, run:  

🔗  [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e50324-9ec0-800b-99cc-c6f59d7e72f1)

```python

# Importing the pandas library
import pandas as pd

# Defining the file path for the CSV file
file_path = '~/playstore_data/Google-Playstore.csv'

# Loading the CSV file into a Pandas DataFrame
df = pd.read_csv(file_path)

# Displaying the first few rows of the dataset to get an overview
print("First few rows of the dataset:")
print(df.head())

# Printing the column names to understand the structure of the dataset
print("\nColumn names in the dataset:")
print(df.columns)


```

🔗  [ChatGPT explanation for the code](https://chatgpt.com/share/67e5033e-8480-800b-9a95-779cef4526b4)


3️⃣  **Click Run (▶) and check the output!** 

 

✅  **Success!** Dataset is now loaded into your notebook and ready for analysis.

## ✍  Step 2: Using **Hugging Face Transformers** for text generation and QA  

This step demonstrates how an LLM can generate text based on a given prompt.

### **➕ 🐍  Add a New Code Cell**    

1️⃣  Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣  Copy and paste the following code:  

🔗  [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e50387-b430-800b-888e-2925f23a8b7a)

3️⃣  import the necessary library:

```python

# Import Hugging Face pipeline

from transformers import pipeline  

```
[ChatGPT explanation for the code](https://chatgpt.com/share/67e503b3-a040-800b-9fe2-9b681d992351)

4️⃣  Load the GPT-2 model, which is a pre-trained language model:

[ChatGPT prompt to generate the code](https://chatgpt.com/share/67e503cf-3fa8-800b-bb16-a1301ceb6602)

```python

# Define text generation task

task = "text-generation"  

# Specify model name

model_name = "gpt2"  

# Load the pre-trained GPT-2 model

generator = pipeline(task, model=model_name)  

```
[ChatGPT explanation for the code](https://chatgpt.com/share/67e503ee-fce8-800b-9f0a-52d3aad73937)


5️⃣  Generate text based on a given prompt:

[ChatGPT prompt to generate the code](https://chatgpt.com/share/67e5040f-3ddc-800b-84e0-f51e0e1bde34)

```python

# Define input prompt

prompt = "Once upon a time, in a futuristic city,"  

# Set maximum number of new tokens

max_tokens = 50  

# Generate text based on prompt

output = generator(prompt, max_new_tokens=max_tokens)  

# Extract generated text

generated_story = output[0]['generated_text']  
# Print generated story

print("Generated Story:")  

print(generated_story)  

```
🔗  [ChatGPT explanation for the code](https://chatgpt.com/share/67e50436-4a68-800b-ba83-8932f401c0f5)

6️⃣  **Click Run (▶) and check the output!** 

  

✅  Text generation complete! You should now see a futuristic story generated by GPT-2. 📖 🚀
🎉 

💡  **Challenge:** Modify the `prompt` variable to explore different stories.

## 🤔  Step 3: Analyze App Descriptions with GPT-2

Instead of manually writing descriptions, let’s use GPT-2 to generate app descriptions 
automatically.

### **➕ 🐍  Add a New Code Cell**    

1️⃣  Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣  Copy and paste the following code:  

🔗  [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e5045a-6430-800b-a98a-9f519ed45f4d)

3️⃣  Extract app names and format them into prompts:

```python

# Extract app names from dataset

app_names = playstore_df['App Name'].dropna()  

# Convert app names into prompts for description generation

app_descriptions = [f"{str(app)} is an app that " for app in app_names]  

```

🔗  [ChatGPT explanation for the code](https://chatgpt.com/share/67e50481-8ba8-800b-adbb-d86d1c9ee51c)


4️⃣  Use GPT-2 to generate descriptions for each app:

🔗  [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e5049f-7f8c-800b-8209-7efdae434384)

```python

# Select first 10 app names to generate descriptions

selected_apps = app_descriptions[:10]  

# Set max tokens for generated descriptions

max_description_tokens = 50  

# Generate text for app descriptions

outputs = generator(selected_apps, max_new_tokens=max_description_tokens)  

```
🔗  [ChatGPT explanation for the code](https://chatgpt.com/share/67e50555-aca8-800b-a92c-32957815cd46)


5️⃣  Print generated app descriptions:

🔗  [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e50573-af9c-800b-a59b-36574c81eb40)

```python

# Print generated descriptions for each app

print("Generated App Descriptions:")  
for app, output in zip(playstore_df['App Name'][:10], outputs):  

    generated_text = output[0]['generated_text']  

    print(f"{app}: {generated_text}")  

```
🔗  [ChatGPT explanation for the code](https://chatgpt.com/share/67e5058e-187c-800b-838d-fd5841958e8a)


6️⃣  **Click Run (▶) and check the output!** 

✅  App descriptions generated! You should now see AI-generated descriptions for the first 10 
apps. 📱 ✨ 🎉 

💡  **Challenge:** Try processing more than 10 apps to analyze the variety of AI-generated text.


---

## 🎯 4. Wrap-Up & Next Steps
🎉 Congratulations! You learned how to:
- ✅ Access **HPC & JupyterLab** 🖥️
- ✅ Install **LLM dependencies on HPC** 🔧
- ✅ Download and analyze **Kaggle datasets** 📊
- ✅ Use **GPT-2 for text generation** ✍️
- ✅ Apply AI to real-world app descriptions 🏗️

🚀 **Next Workshop:** [📚 LLM + RAG (AI-Powered Search)](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/hpc-llm-rag)


### 🔗 Additional Resources 📚
- [Project Jupyter Documentation](https://docs.jupyter.org/en/latest/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>      
- [Microsoft Learn: Introduction to large language models](https://learn.microsoft.com/en-us/training/modules/introduction-large-language-models/)
- [ACCESS CI](https://access-ci.org/get-started/for-educators/) (Free access to HPC for all using the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) U.S. government program) 

🎉 **Keep exploring AI, and see you at the next workshop!** 🚀

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---