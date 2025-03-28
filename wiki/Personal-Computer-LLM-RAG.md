# 🚀✨ **PC: Introduction to Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG)** ✨🚀  


## 🎯 **Goal**  
🤖 **Understand the basics of Large Language Models (LLMs)** and how **Retrieval-Augmented Generation (RAG)** enhances LLMs for tasks like text generation and answering complex queries. You will learn to use LLMs in combination with a retrieval system to improve performance. No prior experience needed—just bring your curiosity! 🚀  

---

## 📌 **What You Will Learn** 🧠💡  
✅ What are **Large Language Models (LLMs)**?  
✅ What is **Retrieval-Augmented Generation (RAG)**?  
✅ Setting Up LLM and RAG for Text Generation  
✅ Running LLM and RAG for Text Generation   
✅ How to use **RAG** with **Hugging Face** to enhance text generation  
✅ Hands-on coding with **Google Colab** and **Hugging Face**  

---

## 🤖 **1. What is a Large Language Model (LLM)?**  
### 🧠 Understanding LLMs in Simple Terms  
A **Large Language Model (LLM)** is a type of AI model that can understand and generate human-like text. It learns by analyzing massive amounts of text data, recognizing patterns, and predicting words based on context.

### 📌 Real-World Examples:  
- ✅ Chatbots like Siri, Google Assistant, ChatGPT 🗣️  
- ✅ AI-powered writing assistants (Grammarly, Jasper AI) ✍️  
- ✅ Search engines predicting your queries 🔍  
- ✅ AI-generated stories and essays 📖  

---

## 🔧 **2. What is Retrieval-Augmented Generation (RAG)?**  

### 🖼️ Retrieval-Augmented Generation (RAG) Visual Representation

<img src="https://github.com/user-attachments/assets/28f1e5d6-070b-405f-8a54-87c17ed5fbab" width="600">

Source: [RAG and LLM Integration](https://apmonitor.com/dde/index.php/Main/RAGLargeLanguageModel)

### 🧠 How RAG Enhances Large Language Models 🔗  

💡 Think: How do retrieval and generation work together to improve AI responses? 🤔

Retrieval-Augmented Generation (RAG) is an AI framework that improves large language models (LLMs) by integrating an external knowledge retrieval process. This allows the model to **pull relevant information from a document database** instead of relying only on its pre-trained knowledge.

- **User Query** (💬): A user submits a question, which is then processed by the system to find relevant information.

- **Vector Database & Document Storage** (📂)
  - Documents are **converted into numerical embeddings** using an **encoder model**.
  - These embeddings are stored in a **vector database** for efficient retrieval.

- **Encoder Model** (🧩)
  - The user's query is transformed into an **embedding representation**.
  - The system finds the closest related documents using **k-Nearest Neighbors (k-NN)**.

- **Context Retrieval & Augmentation** (🔍➡️📖)
  - The **most relevant documents** are retrieved from the vector database.
  - These documents are **added as extra context** for the LLM before generating a response.

- **Large Language Model (LLM) Processing** (🧠)
  - The LLM combines its **pre-trained knowledge** with the **retrieved external information**.
  - This **enhances accuracy and reduces hallucination**, improving response quality.

- **Final Answer Generation** (✅)
  - The model generates a well-informed response using both internal and retrieved knowledge.
  - The final answer is then **returned to the user**.

---

### **Why RAG Matters?** 🚀  
✅ **More Accurate** – Reduces AI hallucinations by retrieving real-time, external information.  
✅ **Scalable** – Works with large document collections without needing to retrain the model.  
✅ **Efficient** – Uses vector search for fast, **semantic** document matching.  

📖Traditional LLMs generate responses based on probability distributions learned from training data. RAG mitigates hallucinations by injecting real-time knowledge, making responses more factually grounded.

### 🛠️ How Does RAG Work?  
RAG enhances LLMs by integrating **retrieval** and **generation** to provide more accurate responses. Instead of relying solely on pre-trained knowledge, it fetches relevant information from external databases before generating a response.

#### 📚 Steps in RAG:  
1️⃣ **Retrieval:** The model searches for relevant documents from a knowledge base.  
2️⃣ **Augmentation:** The retrieved data is passed as context to the LLM.  
3️⃣ **Generation:** The LLM generates a response based on the retrieved information. 

📌 **Example:**  
- If you ask about a recent scientific breakthrough, RAG can retrieve research papers or trusted sources before forming an answer.  

---

## 🔧 **3. Setting Up LLM and RAG for Text Generation**

## 🚀 **Step 1: Open [Google Colab](https://colab.research.google.com/)**

1️⃣ Open your browser and go to **[Google Colab](https://colab.research.google.com/)**.  
2️⃣ Click **+ New Notebook** to begin.

## 🛠️ **Step 2: Set Up Hugging Face Account and Access Token**

1️⃣ **Sign up on Hugging Face**: Go to [Hugging Face Sign-Up](https://huggingface.co/join) and create a free account.  
2️⃣ **Generate an Access Token**:
   - Click on your profile icon and go to **[Your Account Settings](https://huggingface.co/settings/tokens)**.
   - Scroll down to **Access Tokens** and click **New Token**.
   - Give it a name (e.g., "Colab Access") and select **Read** access.
   - Click **Generate Token** and copy the token.

✅ Hugging Face account setup complete! You're now ready to log in. 🔑🎉

## 📚 **Step 3: Login in Colab with the Token**

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4ac60-f484-8011-ba1b-81a3881df8bc)

```python
# Import Hugging Face login module
# This allows secure access to Hugging Face models and datasets
from huggingface_hub import notebook_login  

# Trigger login prompt to authenticate with your access token
notebook_login()  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cafa63-35d0-8004-bfdc-f36ffb25ae57)

3️⃣ **Click Run (▶) and follow the instructions.**  

✅ Logged in successfully! Now, let's verify authentication. 🎉

## 🔐 **Step 4: Verify Authentication**

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4b101-556c-8011-b3e0-994ab46f3ce5)

```python
# Check if authentication is successful
!huggingface-cli whoami  
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67caface-5ed8-8004-a60c-5132ec7113bd)

3️⃣ **Click Run (▶) and check the output!** 
 
✅ If it prints your Hugging Face username, the setup is complete! 🎉


---

## 🔧 **4. Running LLM and RAG for Text Generation**

## 📚 **Step 1: Install and Import Required Libraries**

Before importing the libraries, install the necessary dependencies by running the following command:

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4b1b4-1b80-8011-89f0-28f7dc04c5df)

```python
# Install Hugging Face Transformers, FAISS, and datasets
# FAISS is used for efficient similarity search and clustering of dense vectors
# Datasets is needed to create and manage document collections for retrieval
!pip install transformers faiss-cpu datasets torch
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cafb46-9940-8004-8618-4f0cb0dfd5aa)

3️⃣ **Click Run (▶) to install the required packages.**

✅ Dependencies installed successfully! Now, let's import the necessary libraries. 📚🎉

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4b230-42d0-8011-9228-f76a2c886b12)

```python
# Import required libraries for RAG implementation
from transformers import RagTokenizer, RagRetriever, RagSequenceForGeneration
from datasets import load_dataset
import torch

# For visualization and debugging
import pandas as pd
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67cafbb3-4074-8004-8fd6-58b28b1cdd4d)

3️⃣ **Click Run (▶) to import the libraries.**

✅ Libraries imported successfully! You're now ready to set up the knowledge base for retrieval. 🚀🎉

## 🗃️ **Step 2: Set Up Knowledge Base for Retrieval**

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4b76e-af5c-8011-9981-c235b7717aec)

```python
# Load a small dataset to use as our knowledge base
# We're using the "wiki_qa" dataset which contains question-answer pairs
print("Loading knowledge base dataset...")
dataset = load_dataset("wiki_qa", split="train")

# Let's look at what our knowledge base contains
print("\nDataset structure:")
print(dataset)

# Let's see a few examples from our knowledge base
print("\nSample entries from our knowledge base:")
for i in range(3):  # Show 3 examples
    print(f"\nEntry {i+1}:")
    print(f"Question: {dataset[i]['question']}")
    print(f"Answer: {dataset[i]['answer']}")
    print(f"Document: {dataset[i]['document_title']}")
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e4b726-2994-8011-b6dd-fc6bcad1ab65)

3️⃣ **Click Run (▶) to set up the knowledge base.**

✅ Knowledge base set up successfully! Now, let's load the RAG model components. 🧠🎉

## 🧠 **Step 3: Load the RAG Model Components**

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4b819-c904-8011-b8cc-9b0c46b8bb6c)

```python
# Load the tokenizer for processing text input
print("Loading RAG tokenizer...")
tokenizer = RagTokenizer.from_pretrained("facebook/rag-sequence-nq")

# Set up the retriever component
# The retriever is responsible for finding relevant documents in our knowledge base
print("\nLoading RAG retriever...")
retriever = RagRetriever.from_pretrained(
    "facebook/rag-sequence-nq",
    index_name="custom",  # We're using a custom index
    use_dummy_dataset=True  # For demonstration purposes
)

# Load the complete RAG sequence generation model
# This model combines the retriever with a generator to produce answers
print("\nLoading RAG sequence generation model...")
model = RagSequenceForGeneration.from_pretrained("facebook/rag-sequence-nq")

print("\nRAG model components loaded successfully!")
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e4b7c2-2154-8011-a43d-28fe6e40de07)


3️⃣ **Click Run (▶) to load the RAG model components.**

📌 **Note:** When running this step, you may see a prompt asking:

```
Do you wish to run the custom code? [y/N]
```

Type **'y'** and press **Enter** to allow the model to load properly. This is required for some Hugging Face models.

✅ RAG model components loaded successfully! Now, let's prepare a query and generate a response using RAG. 🚀🎉

## 📚 **Step 4: Prepare a Query and Generate a Response with RAG**

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4b892-5de4-8011-b0d9-b301ab8ed253)

```python
# Define a function to generate responses using our RAG system
def generate_rag_response(question):
    # Encode the question for the model
    input_ids = tokenizer.encode(question, return_tensors="pt")
    
    # Generate a response
    # The model will retrieve relevant documents and use them to create an answer
    print(f"Generating response for: '{question}'")
    outputs = model.generate(
        input_ids,
        max_length=150,  # Adjust this for longer/shorter responses
        num_beams=5,      # Beam search for better quality outputs
        early_stopping=True
    )
    
    # Decode and return the generated response
    generated_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    
    return generated_text

# Define our query - feel free to change this to any question you like!
query = "What is the process of photosynthesis?"

# Generate a response using our RAG system
response = generate_rag_response(query)

# Display the response
print("\n✨ Generated Answer:")
print(response)
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e4b857-5190-8011-937f-f2243ad226c1)

3️⃣ **Click Run (▶) to generate a response using the complete RAG system.**

🎯 **Challenge: Try your own questions!**
- Edit the `query = "What is the process of photosynthesis?"` line with your own question
- Try questions about historical events, scientific processes, or general knowledge
- Compare the responses with what you know about the topics
- See how the model's responses are informed by the retrieved information

✅ RAG response generated successfully! You now have a complete RAG system retrieving information and generating answers. 🚀🎉

## 🔍 **Step 5: Visualize the Retrieval Process (Optional)**

### **➕🐍 Add a New Code Cell**  
1️⃣ Click **+ Code** in the top left to add a new code cell.  
2️⃣ Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4b93d-8364-8011-b620-4210763c0f7b)

```python
# This function helps us see what documents were retrieved for a query
def inspect_retrieved_documents(question):
    # Prepare the query
    input_dict = tokenizer.prepare_seq2seq_batch(
        question,
        return_tensors="pt"
    )
    
    # Get document IDs retrieved for the question (this is simplified)
    # In a full implementation, we would extract the actual document IDs
    print(f"Retrieval process for query: '{question}'")
    print("\nIn a complete RAG system, here we would see:")
    print("1. The question being transformed into an embedding")
    print("2. Similar documents being found using vector similarity")
    print("3. Retrieved documents being passed to the model with the question")
    
    # Display a visual representation of the RAG process
    print("\n📊 RAG Pipeline Visualization:")
    print("Query → [Encoder] → Query Embedding")
    print("                       ↓")
    print("Knowledge Base → [Retriever] → Relevant Documents")
    print("                                 ↓")
    print("Query + Relevant Documents → [Generator] → Enhanced Response")

# Let's see how retrieval works for our query
inspect_retrieved_documents(query)
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e4b8dc-e824-8011-a867-22c3b3e7a0a7)

3️⃣ **Click Run (▶) to visualize the retrieval process.**

✅ You now understand how the retrieval process works in a RAG system! 🔍🎉

---

## 🎯 **5. Wrap-Up & Next Steps**  
🎉 Congratulations! You learned how to:  
✅ Use a **Large Language Model (LLM)** for text generation.  
✅ Implement **Retrieval-Augmented Generation (RAG)** to improve text generation.  
✅ Use **Hugging Face** for easy access to pre-trained models and retrieval systems.  

🚀 **Next Workshop:** [🔍 Ethical AI & Future Trends](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/personal-computer-ethical-ai)  

### 🔗 **Additional AI Resources** 📚   

- [Google Colab Guide](https://colab.research.google.com/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>       
- [Microsoft: AI for Beginners](https://microsoft.github.io/AI-For-Beginners/?id=other-curricula)
- [Microsoft: RAG and Knowledge Retrieval Fundamentals](https://learn.microsoft.com/en-us/shows/rag-time-ai-learning/rag-and-knowledge-retrieval-fundamentals)


🎉 Keep learning AI, and see you at the next workshop! 🚀

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---
