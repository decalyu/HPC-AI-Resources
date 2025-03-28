
# 📌 Intro to Command Terminal, AI, & Jupyter – A Beginner’s Guide 🖥️🤖📓  

Welcome to your **AI Journey!** 🚀 This workshop is designed for **complete beginners**—no experience needed! By the end, you'll know how to **talk to your computer**, **explore AI**, and **write your first code in Jupyter Notebook**!  

---

## 🎯 What You’ll Learn
- **Command Terminal Basics** – Talk to your computer with text commands.
- **AI 101** – What AI is and how it works in everyday life.
- **Jupyter Notebook** – A fun way to write and run code.

**No coding experience? No problem!** Let’s start from the basics. 🌟  

---

## 🛠️ What You’ll Need
✅ A computer (Windows, macOS, or Linux). 💻  
✅ Internet access (for hands-on activities). 🌍  
✅ **No installations required!** We’ll use an online tool first.  
✅ **A sense of curiosity!** 🧠✨  

---

# 🔹 Lesson 1: Talking to Your Computer (Command Terminal Basics) ⌨️  

The **command terminal** is like a **secret code** that lets you control your computer with text commands instead of clicking buttons.  

### 📌 How to Open the Terminal  
#### Windows (Use Command Prompt)  
1. Press **Win + R**, type `cmd`, and press **Enter**.  
   *You’ll see a black window pop up. This is the command prompt! It’s like your computer’s "chat" window where you type commands.*

#### macOS  
1. Press **Cmd + Space** to open **Spotlight Search**.  
2. Type `Terminal` and press **Enter**.  
   *The terminal looks like a black box, but don’t worry—it's where you will tell your computer what to do.*

#### Linux  
1. Press **Ctrl + Alt + T** to open the terminal.  
   *Now you’re ready to start talking to your computer!*

---

### 📌 Try Your First Commands!  

#### What does each command do?

1. **`whoami`** – This tells the terminal your computer username.
   
   ```bash
   whoami
   ```

2. **`echo "Hello, AI World!"`** – This command lets you send a message to the terminal, like a chat. Here, it says "Hello, AI World!".
   
   ```bash
   echo "Hello, AI World!"
   ```

3. **`pwd`** – This shows you **where you are** inside your computer. It’s like checking what room you’re in.
   
   ```bash
   pwd
   ```

🎉 Congratulations! You just used the **command terminal** like a pro.  

---

# 🔹 Lesson 2: AI in Everyday Life 🤖  

### 📌 What is AI?  
AI stands for **Artificial Intelligence**—a technology that helps computers **think and learn** like humans!  

🔍 **Examples of AI You Use Every Day:**  
✅ **Chatbots** – AI that answers your questions (e.g., ChatGPT). 💬  
✅ **Face Unlock** – AI recognizes your face to unlock your phone. 📱  
✅ **Google Translate** – AI translates languages in real time. 🌍  
✅ **Netflix & Spotify** – AI recommends movies and music you’ll like. 🎶🎬  

### 📌 Interactive AI Demo (No Coding!)  
Let’s see AI in action! Click the link below to play with **Google’s Quick, Draw!**, an AI that recognizes drawings.  

👉 [**Try Quick, Draw!**](https://quickdraw.withgoogle.com/)  

🚀 AI learns from **millions of drawings**—try drawing an object and see if AI can guess it!  

---

# 🔹 Lesson 3: Running Your First AI Code in Jupyter Notebook 📓✨  

### 📌 What is Jupyter Notebook?  
Jupyter Notebook is a **digital notebook** where you can write and run code, see results instantly, and even make charts and graphs! It’s the perfect place for beginners to explore Python and AI.

---

### 🚀 Step 1: Open Jupyter Notebook (Google Colab)  
👉 [Click Here to Open Google Colab](https://colab.research.google.com/)  

1. Click **"New Notebook"**.  
2. You’ll see a blank cell where you can write code. This is where the magic happens!  

---

### 🚀 Step 2: Run Your First Python Code!  

Python is the language we’ll use to write code. Let’s start with a **simple command** to make the computer say “Hello, AI World!”

#### Copy and paste this into the first cell:

```python
print("Hello, AI World!")
```

Now press **Shift + Enter** to run it! 🎉

🔹 **Modify the Code:** Change `"Hello, AI World!"` to `"Hello, [Your Name]!"` and run it again to make it personal! 📝  

---

# 🔹 Lesson 4: AI in Action – Let’s Try Image Recognition! 📸  

### 📌 Step 1: Load a Pre-Trained AI Model  
A **pre-trained model** is like a brain that’s already learned to recognize things, like objects in pictures. Let's load one that can recognize images.

#### Copy and paste this code into a new cell in Google Colab:

```python
from tensorflow.keras.applications import MobileNetV2  
from tensorflow.keras.preprocessing import image  
import numpy as np  

model = MobileNetV2(weights='imagenet')  

print("AI model loaded! Ready to recognize images.")  
```

Press **Shift + Enter** to run it. 🎉  
You’ve just loaded an AI model that can recognize objects in images!  

---

### 📌 Step 2: Try Recognizing an Image  

Now, let's ask our AI to recognize an image of a **cat**. It will predict what the object is!

#### Copy and paste this code into the next cell:

```python
from tensorflow.keras.applications.mobilenet_v2 import decode_predictions  

# Load an example image
img_path = 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg'
img = image.load_img(img_path, target_size=(224, 224))  
img_array = image.img_to_array(img)  
img_array = np.expand_dims(img_array, axis=0)  

# Make a prediction
predictions = model.predict(img_array)  
label = decode_predictions(predictions)  

print("AI thinks this is:", label[0][0][1])  
```

Run the code and see what the AI predicts! 🐱

---

# 🔹 What’s Next? 🚀  

✅ **You’ve learned:**  
✔️ How to use the **command terminal**  
✔️ What AI is and how it’s used in real life  
✔️ How to write and run **Python code in Jupyter Notebook**  
✔️ How to use an **AI model** to recognize images  

🔹 **Next Steps:**  
📊 **[Explore Data & Build Your Own AI Model!](personal-computer-data-exploration)**  

🙋 **Have Questions?** Drop them below! 💡  

---
