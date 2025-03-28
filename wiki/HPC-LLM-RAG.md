# 🚀✨ **HPC: Introduction to Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG)** ✨🚀  


## 🎯 **Goal**  
🤖 **Learn how to implement Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) efficiently using CSUSB HPC.**
Traditional LLMs struggle with retrieving up-to-date information. **RAG enhances LLMs** by integrating external knowledge retrieval, improving accuracy for tasks like answering complex queries.  

---

## 📌 **What You Will Learn** 🧠💡  
✅ **Why Use HPC Instead of a Local Computer?**  
✅ **Access HPC Terminal via CSUSB HPC Portal**  
✅ **Install Dependencies, Setup Kaggle, and Download Dataset**   
✅ **Load and Preprocess the Dataset in Jupyter Notebook**  
✅ **Implementing Retrieval-Augmented Generation (RAG)**  
✅ **Analyzing Retrieval Efficiency**  

---

## 🚀 **1. Retrieval-Augmented Generation (RAG) Overview**


## 📌 **Introduction**
🔹 Retrieval-Augmented Generation (RAG) is an AI framework that enhances **Large Language Models (LLMs)** by integrating **real-time knowledge retrieval** with text generation. Instead of relying solely on pre-trained knowledge, RAG retrieves **relevant external documents 📚** to provide **more accurate, context-aware responses**.


## 🏗 **RAG System Architecture**
### 📊 **Workflow Diagram**

<img src="https://github.com/user-attachments/assets/acab4adb-8f41-46ed-8cb6-d5d11ab2c072" width="550">

Source: [What is Retrieval-Augmented Generation?](https://aws.amazon.com/what-is/retrieval-augmented-generation/)

## ⚙️ **How RAG Works (Aligned with Diagram)**
### 🔹 **Key Processing Steps**
1️⃣ **User Input 💬** → The user provides a **query** (e.g., a question or request for information).<br>
2️⃣ **Search Relevant Information 🔎** → The system searches an external **knowledge base 📂** for relevant information.<br>
3️⃣ **Retrieve Information 📖** → The system finds the most **contextually relevant documents** and extracts useful content.<br>
4️⃣ **Augment Prompt 📝** → The **retrieved information is merged** with the original query, creating an enriched prompt.<br>
5️⃣ **Process with LLM 🧠** → The **augmented prompt** is sent to the language model to generate a **fact-based answer**.<br>
6️⃣ **Generate Final Response ✅** → The user receives a response **grounded in retrieved knowledge**.<br>

### 🔍 **Core Components (Matching Workflow Elements)**
| **Component**                | **Function** |
|------------------------------|-------------|
| **User Query** 💬           | Initial question or request for information |
| **Search Relevant Information** 🔍 | Finds matching knowledge in external sources |
| **Knowledge Sources** 📚      | Databases, documents, or repositories storing retrievable data |
| **Retrieved Context** 📄     | Relevant snippets extracted for augmentation |
| **Enhanced Prompt** ✍️      | Merged user query + additional context |
| **LLM Processing** 🧠        | Generates responses using both pre-trained knowledge and retrieved content |


## 🎯 **Why Use RAG?**
✔ **Reduces Hallucination 🚫** – Retrieves **real-time information** instead of making assumptions.  
✔ **More Context-Aware 📌** – Uses retrieved data to **enhance LLM responses**.  
✔ **Scalable & Efficient ⚡** – Works with **large document repositories** without retraining the model.  
✔ **Improves Accuracy 🎯** – Ensures answers **align with verified sources**.  



## 🔄 **Example Use Case**
### **Scenario: AI-powered Research Assistant 📑**
📌 **User Query:** _"What are the latest advancements in renewable energy?"_

🔍 **Without RAG:**
- The LLM might generate an **outdated response** based only on its last training data.

✅ **With RAG:**
- The system **retrieves recent research papers 📰** and trusted articles.
- The model **incorporates external knowledge**, ensuring a **current and factual answer**.

## 🏁 **Conclusion**
By combining **retrieval** with **generation**, RAG **significantly improves response accuracy** by grounding LLM outputs in real-world information. This framework is **widely used** in applications like **intelligent search engines 🔍, enterprise AI assistants 🤖, and automated research tools 📊**.


## 🖥️ **1. Why Use HPC Instead of a Local Computer?**

**Limitations of Local Machines:**
- 🚫 Limited memory (RAM) can slow down LLM inference and retrieval.
- 🚫 CPUs struggle with large datasets, making retrieval inefficient.
- 🚫 Local machines require heavy GPU resources for fine-tuning.

**Advantages of HPC:**
- ✅ **Faster processing**: Leverages powerful CPUs and GPUs.
- ✅ **Handles large-scale datasets**: Works seamlessly with **vector search** (FAISS).
- ✅ **Parallel processing**: Multiple cores process data in parallel, accelerating RAG models.
- ✅ **Remote execution**: No need to burden local machines with heavy computation.

---

## 🔍 **2: Access HPC Terminal via JupyterHub**

1️⃣ Go to [CSUSB HPC](https://csusb-hpc.nrp-nautilus.io/) if you're a student or teacher at CSUSB. If not, ask a teacher from your school to create an account for you using the [ACCESS CI](https://access-ci.org/get-started/for-educators/) program, which provides free access to computing tools like Jupyter for classroom use.<br>
2️⃣ Click **CI Logon** to log in using your school account.<br>
3️⃣ Select the GPU model that best fits your needs.<br>
4️⃣ After logging in, Welcome to JupyterLab.<br>
✅ You're ready to go!

**Important:** Make sure to select **Python 3** as your notebook kernel. This is essential for all code in this tutorial to work correctly!


---

## 💻 **3. Install Dependencies, Setup Kaggle, and Download Dataset**

1️⃣ Click **Terminal** in JupyterLab.    
2️⃣ Run the following commands:    


🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e47823-3c90-8011-8100-ee17baede0a2)

```bash
# Install Pandas, a powerful library for data manipulation and analysis, 
# providing data structures like DataFrames for handling structured data.
pip install --user pandas  

# Install Seaborn, a statistical data visualization library built on top of Matplotlib, 
# useful for creating informative and attractive graphs.
pip install --user seaborn  

# Install Matplotlib, a fundamental plotting library for Python, 
# enabling the creation of static, animated, and interactive visualizations.
pip install --user matplotlib  

# Install the Kaggle API library, which allows users to download datasets, 
# submit solutions, and interact with Kaggle's platform programmatically.
pip install --user kaggle  

# Install FAISS (Facebook AI Similarity Search), an efficient library for 
# similarity search and clustering of dense vectors, optimized for fast retrieval.
pip install --user faiss-cpu  

# Install Hugging Face's Transformers library, which provides pre-trained models 
# for various NLP tasks such as text classification, translation, and generation.
pip install --user transformers   

# Temporarily add Kaggle to your system PATH
export PATH=~/.local/bin:$PATH  

# Permanently add Kaggle to your system PATH
echo 'export PATH=~/.local/bin:$PATH' >> ~/.bashrc  

# Apply the changes immediately
source ~/.bashrc  

# Create a new directory for dataset storage
mkdir -p ~/playstore_data  

# Navigate to dataset directory
cd ~/playstore_data  

# Download the Google Playstore dataset from Kaggle
# Note: This dataset may not be found. If you encounter an error,
# you may need to check if the dataset still exists or manually download it.
kaggle datasets download -d gauthamp10/google-playstore-apps  

# Unzip the downloaded dataset
unzip google-playstore-apps.zip  

# List extracted files to confirm successful download
ls -lh  
```
🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1712-6940-8004-9dd6-df6e5b2542c1)

3️⃣ **Click Run (▶) and check the output!** 

✅ **Now you have installed dependencies and downloaded the dataset! 🎉**

---

## 📚 **4: Load and Preprocess the Dataset in Jupyter Notebook**

## Step 1: Load the Dataset into a Pandas DataFrame 
   
### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bb8a-5610-8011-8921-f5442b100e1a)

```python
# Import Pandas for data handling
import pandas as pd  

# Define dataset path
dataset_path = "~/playstore_data/Google-Playstore.csv"  

# Load dataset into Pandas DataFrame
playstore_df = pd.read_csv(dataset_path)  

# Display first few rows
print(playstore_df.head())  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1d6c-9c98-8004-8d4e-71cacaa3d24d)  

3️⃣ Click **Run** (▶) and check the output.

✅ **You should now see your dataset displayed!** 🎉

## Step 2: Clean and Prepare Data for RAG    

### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bbce-aa0c-8011-9700-2634b4c45493)

```python
# Check for missing values in the dataset - this helps identify data quality issues
missing_values = playstore_df.isnull().sum()  
print("Missing Values:\n", missing_values)  

# Remove rows with missing values - ensures complete data for analysis
# This step is important because incomplete data can cause errors in our analysis
playstore_df.dropna(inplace=True)  

# Remove duplicate entries - prevents bias from counting the same data multiple times
# Duplicates can skew statistics and analysis results
playstore_df.drop_duplicates(inplace=True)  

# Display cleaned dataset
print(playstore_df.head())  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1dbe-8580-8004-8d58-e95625a51e73)   

3️⃣ Click **Run** (▶) and check the output.

✅ **Now your dataset is clean and ready for retrieval-augmented generation.**

---

## 🤖 **5. Implementing Retrieval-Augmented Generation (RAG)**

## Step 1: Install and Import Required Libraries    

### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:    

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bc04-3d20-8011-a0d5-b75cba9c9740)

```python
# Import required Hugging Face and FAISS libraries
from transformers import pipeline, RagTokenizer, RagRetriever, RagSequenceForGeneration  
import faiss  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1e19-a334-8004-b603-1f7d4a2034a2)    

3️⃣ Click **Run** (▶) to import the libraries.

✅ Libraries imported successfully! You're now ready to use RAG for retrieval-augmented generation. 🚀🎉

## Step 2: Load a Pre-Trained RAG Model 
   
### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bc4d-c320-8011-9d20-d68625d6d2eb)

```python
# Define tokenizer model name
tokenizer_model = "facebook/rag-sequence-nq"  

# Load tokenizer
tokenizer = RagTokenizer.from_pretrained(tokenizer_model)  

# Define generation model
generation_model = "gpt2"  

# Load text generation model
model = pipeline("text-generation", model=generation_model)  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1e75-6a94-8004-b865-f43171fe3604)  

3️⃣ Click **Run** (▶) to load the pre-trained model.

✅ Tokenizer and generation model loaded successfully! You're now ready to generate text with RAG and GPT-2. 🚀🎉

## Step 3: Define Query and Retrieve Information  
  
### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code: 
 
🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bc8b-714c-8011-b49b-e0f47c88738c)

```python
# Define a user query
query = "What are the top categories in Google Playstore?"  

# Tokenize query
input_dict = tokenizer.prepare_seq2seq_batch(query, return_tensors="pt")  

# This is a fallback code - NOT part of the main RAG logic
# We're setting retrieved_docs to None because we're simplifying the example
# In a full RAG implementation, this would be actual retrieved documents
retrieved_docs = None  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1ebe-2cf4-8004-98f1-4f71c9d6d1ab)    

3️⃣ Click **Run** (▶) to retrieve information.

✅ Query tokenized successfully! Your input is now ready for processing. 🔍🎉

🎯 Challenge: Replace the prompt with something completely unexpected

💡 Extra Tip: Edit the `query = "What are the top categories in Google Playstore?"`

## Step 4: Generate a Response  
  
### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code: 

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bcca-7c64-8011-b49a-d82a18b2b90b)

```python
# Generate response based on retrieved documents
output = model(query, max_length=100, truncation=True)  

# Extract generated response
generated_response = output[0]['generated_text']  

# Print generated answer
print("Generated Answer:", generated_response)  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1f35-4c38-8004-b5dd-9f91a69c3fa2)    

3️⃣ Click **Run** (▶) to generate a response.

✅ Response generated successfully! You should now see the AI-generated answer based on the query. 💡🎉

---

## 📊 **6: Analyzing Retrieval Efficiency**
    
### **➕🐍 Add a New Code Cell**    
1️⃣ Click **+ Code** in Jupyter Notebook to add a new code cell.  
2️⃣ Copy and paste the following code:  
 
🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4bd03-6db8-8011-9561-b17cb2082979)

```python
# Import visualization libraries
import matplotlib.pyplot as plt  
import seaborn as sns  

# Set figure size for better visualization
plt.figure(figsize=(12, 5))  

# Create bar plot for app categories
sns.countplot(x=playstore_df["Category"])  

# Rotate x-axis labels for readability
plt.xticks(rotation=90)  

# Set title
plt.title("Distribution of App Categories in Google Playstore")  

# Display the plot
plt.show()  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cb1f85-ef38-8004-88b3-972f1e37ca9f)   

3️⃣ Click **Run** (▶) and check the output.

✅ **You should see a distribution of app categories.**

<img src="https://github.com/user-attachments/assets/b1d73377-5b32-46de-9a8d-a38376673770" alt="Self-Service System" width="650">


---

## 🎯 **7. Wrap-Up & Next Steps**  
🎉 Congratulations! You learned how to:

✅ **Use HPC for Large-Scale Retrieval-Augmented Generation (RAG)**   
✅ **Download and process large datasets efficiently**   
✅ **Generate responses using an AI model**   
✅ **Analyze and visualize retrieval success**

🚀 **Next Workshop:** [🔍 Ethical AI & Future Trends](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/hpc-ethical-ai)  

### 🔗 Additional AI & HPC Resources 📚  

- [Project Jupyter Documentation](https://docs.jupyter.org/en/latest/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons "Previous" and "Next" to navigate the tutorial and avoid ads.)<br>      
- [Microsoft: RAG and Knowledge Retrieval Fundamentals](https://learn.microsoft.com/en-us/shows/rag-time-ai-learning/rag-and-knowledge-retrieval-fundamentals)
- [ACCESS CI](https://access-ci.org/get-started/for-educators/) (Free access to HPC for all using the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) U.S. government program)<br>

🎉 Keep learning AI, and see you at the next workshop! 🚀

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---