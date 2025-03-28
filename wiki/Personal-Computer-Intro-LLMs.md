# 🚀 ✨  PC: Introduction to Large Language Models (LLMs) ✨ 🚀 

## 🎯  Goal

🤖  Understand what a Large Language Model (LLM) is and how it can be used for text 
generation, question answering, and more, using Python. No prior experience needed—just 
bring your curiosity! 🚀 

---

## 📌  What You Will Learn 🧠 💡 

✅  What is a Large Language Model (LLM)?  
✅  How do LLMs work?  
✅  How to use pre-trained LLMs for text generation  
✅  How to use LLMs for answering questions  
✅  Hands-on coding with Google Colab  
✅  Basics of Hugging Face library for LLMs  

---

## 🤖  1. What is a Large Language Model (LLM)?

### 🧠  Understanding LLMs in Simple Terms

A **Large Language Model (LLM)** is a type of AI model that can understand and generate 
human-like text. It learns by analyzing huge amounts of text data, recognizing patterns, and 
predicting words based on context.

### 📌  Real-World Examples:

- ✅  Chatbots like Siri, Google Assistant, ChatGPT 🗣   

- ✅  AI-powered writing assistants (Grammarly, Jasper AI) ✍   

- ✅  Search engines predicting your queries 🔍   

- ✅  AI-generated stories and essays 📖   

---

## 🔥  2. How do LLMs work?

### 🛠  How Do LLMs Generate Text?

- #### 📚  Training on Large Datasets  

  - LLMs are trained on **billions of words** from books, articles, and websites.

- #### 🔍  Predicting the Next Word  

  - When given a sentence, the model predicts the most likely next word.
- #### ✍  Generating Coherent Responses  

  - By repeating this process, it forms complete sentences that make sense.

📌  **Example: How AI predicts text**   

- Input: "The sky is"    

- Predicted Output: "blue because of the way light scatters."

💡  *Quick Thought: Where else do you see AI predicting text? Jot down your ideas! 📝 *

---

## 🔧  3. How to use pre-trained LLMs for text generation

## 💻  Step 1: Open Google Colab

1️⃣  Open [Google Colab](https://colab.research.google.com/)  
2️⃣  Click **+ New Notebook**  

## 📚  Step 2: Import Required Libraries

### **➕ 🐍  Add a New Code Cell**         

1️⃣  Click **+ Code** in the top left to add a new code cell.  
2️⃣  Copy and paste the following code into the new code cell.  

🔗 [ChatGPT prompt for generating the code](https://chatgpt.com/share/67e505fd-7298-800b-a710-446755328d7b) 
```python

# Import the pipeline function from the transformers library
from transformers import pipeline

# Load a pre-trained text generation pipeline
# The 'text-generation' pipeline uses a model like GPT-2 by default
text_generator = pipeline("text-generation")

# Define a prompt for text generation
prompt = "Once upon a time, in a distant kingdom,"

# Generate text based on the prompt
# The max_length parameter controls the output length
output = text_generator(prompt, max_length=50)

# Print the generated text
print(output)
  

```
🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e50abd-4408-800b-9f23-48e5c7f03265)

3️⃣  **Click Run (▶) and check the output!** 

✅  Libraries imported successfully! You’re now ready to use the Hugging Face pipeline. 🚀 🎉 

## 🧠  Step 3: Load a Pre-Trained LLM

💡  We will use GPT-2, a popular model that generates human-like text.  

### **➕ 🐍  Add a New Code Cell**           

1️⃣  Click **+ Code** in the top left to add a new code cell.  
2️⃣  Copy and paste the following code into the new code cell.  

🔗  [ChatGPT prompt for generating this code](https://chatgpt.com/share/67e507c7-a58c-800b-b183-3ec8627ee76a)
```python

# First, we need to install the transformers and torch libraries if they are not already installed.
# Uncomment and run the following line if you haven't installed them:
# !pip install transformers torch

# Import necessary libraries
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load the pre-trained GPT-2 model and tokenizer
# GPT2Tokenizer is used for encoding input text and GPT2LMHeadModel is used for text generation.

# Load pre-trained GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Load pre-trained GPT-2 model
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Step 2: Encode the input text
# Tokenize the input text using the tokenizer. 
# The model expects input in tokenized form, so the text is converted into token IDs.
input_text = "Once upon a time, there was a brave knight"
input_ids = tokenizer.encode(input_text, return_tensors='pt')

# Step 3: Generate text using the model
# The model generates a continuation of the input text based on the provided input.
# Here, 'max_length' specifies the maximum length of the generated text.
output = model.generate(input_ids, max_length=100, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.95, temperature=0.7)

# Step 4: Decode the generated output
# After the model generates the token IDs, we decode them back to text.
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

# Step 5: Display the generated text
print("Generated text:\n", generated_text)


```
🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e50881-bee8-800b-9a21-45bd0cb18b95)

3️⃣  **Click Run (▶) and check the output!** 

✅  Pre-trained LLM loaded successfully! You're now ready to generate text using GPT-2. 🧠
🚀 🎉 

## 📝  Step 4: Generate Text Using the LLM    

### **➕ 🐍  Add a New Code Cell**         

1️⃣  Click **+ Code** in the top left to add a new code cell.  
2️⃣  Copy and paste the following code into the new code cell.  

🔗  [ChatGPT prompt for generating this code](https://chatgpt.com/share/67e508c2-7e60-800b-a9ea-91b48ed3d183)

```python

# Install the necessary libraries (uncomment if not installed)
# !pip install transformers torch

# Import required libraries from Hugging Face's transformers
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Define the input text prompt
input_prompt = "Once upon a time in a land far away"

# Set the maximum length for the generated text
max_length = 100  # You can adjust this to control the length of the output text

# Load the pre-trained GPT-2 model and tokenizer
# GPT-2 is a popular language model trained by OpenAI
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Encode the input text prompt into tokens that the model understands
# This converts the input string into a format that can be fed into the model
input_ids = tokenizer.encode(input_prompt, return_tensors='pt')

# Generate text using the model based on the input prompt
# 'max_length' specifies the maximum length of the output sequence (including the input prompt)
generated_output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, temperature=0.7)

# Decode the generated tokens back into human-readable text
# This step converts the token IDs back into a string
generated_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)

# Print the generated text
print("Generated Text:\n", generated_text)

```
🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e508e5-540c-800b-a5fa-9a16b438b3ef)

3️⃣  **Click Run (▶) and check the output!** 

✅  Text generated successfully! You should now see a creative continuation of your prompt. 
📝 ✨ 🎉 

🎯  Challenge: Try changing the prompt to something funny or mysterious!🔥 <br>

💡  Extra Tip: Edit the prompt_text = "..." line and try your own creative ideas. For example, 
start with “In a haunted bakery...” or “The cat who ruled the internet...”

---
## 🤖  4. How to use LLMs for answering questions

💡  LLMs can also answer questions based on a given context!  

### **➕ 🐍  Add a New Code Cell**           

1️⃣  Click **+ Code** in the top left to add a new code cell.  
2️⃣  Copy and paste the following code into the new code cell. 

 

🔗  [ChatGPT prompt for generating this code](https://chatgpt.com/share/67e50bac-1818-800b-837b-f8c83ed4681f)

```python

# Step 1: Install necessary libraries (if not installed yet)
# You can run these commands in your terminal or Jupyter notebook cell
# !pip install transformers torch

# Step 2: Import necessary libraries
from transformers import pipeline

# Step 3: Load the question-answering pipeline using the 'distilbert-base-uncased-distilled-squad' model
qa_pipeline = pipeline("question-answering", model="distilbert-base-uncased-distilled-squad")

# Step 4: Define the context (the passage that contains information to answer the question)
context = """
Hugging Face is a company that focuses on Natural Language Processing (NLP). They have developed a popular library called transformers, 
which provides pre-trained models for various NLP tasks, including text generation, sentiment analysis, and question answering. 
Hugging Face aims to democratize AI by making these models accessible to the wider community.
"""

# Step 5: Define the question you want to ask
question = "What does Hugging Face do?"

# Step 6: Use the QA pipeline to answer the question based on the context
answer = qa_pipeline(question=question, context=context)

# Step 7: Print the answer
print(f"Question: {question}")
print(f"Answer: {answer['answer']}")

```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67e50bcc-dfdc-800b-8776-e8b619220e63)


3️⃣  **Click Run (▶) and check the output!** 

✅  Question answered successfully! You should now see the model's response based on the 
given context. 🤖 🎉 

---

## 🎯  5. Wrap-Up & Next Steps

🎉  Congratulations! Today you learned how to:

- ✅  Use LLMs for text generation ✍ 
- ✅  Use LLMs for question answering 💬 
- ✅  Work with Hugging Face models 🤖 

🚀  **Next Workshop:** [📚  LLM + RAG (AI-Powered Search)](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/personal-computer-llm-rag)

### 🔗  Additional AI Resources 📚 

- [Google Colab Guide](https://colab.research.google.com/)     

- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two 
green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br> 

- [AI for Beginners (Microsoft)](https://microsoft.github.io/AI-For-Beginners/?id=other-curricula)

- [What is LLM (Large Language Model)?](https://aws.amazon.com/what-is/large-language-model/)

🎉  Keep learning AI, and see you at the next workshop! 🚀 

---

### 📝  Workshop Feedback Survey 

Thanks for completing this workshop!🎆 

We'd love to hear what you think so we can make future workshops even better. 💡 

📌  **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---
