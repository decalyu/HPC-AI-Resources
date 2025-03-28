# 🚀✨ **PC: 📊 Visualize AI Data!** ✨🚀

## 🎯 **Goal**

📊 Learn how to **clean, analyze, and visualize** real-world datasets using Python tools like Pandas and Seaborn. No prior experience needed—just bring your **curiosity and creativity**! 🚀

---

## 📌 What You Will Learn 🧠💡

✅ What datasets are and why they matter 📂<br>
✅ How libraries help process and visualize data 🛠️<br>
✅ Loading and viewing data in Google Colab 🔍<br>
✅ Handling missing values and formatting datasets 🧹<br>
✅ Using statistics to understand datasets 📊<br>
✅ Organizing data for better insights 📈<br>
✅ Creating charts and graphs for analysis 🎨

---

## 📚 **1. Understanding Datasets 🤔**

### **What is a Dataset? 📂**

A **dataset** is a collection of structured information that AI models use to learn. Data can come in different formats, such as:

📸 **Images** (e.g., photos of animals for recognition)  
📜 **Text** (e.g., social media posts for chatbots)  
📊 **Numbers** (e.g., weather reports for predictions)  
🎵 **Audio** (e.g., music for recommendations)  

💡 **Think:** Where else do you see AI using datasets in real life? 🤔💭

---

## 📚 **2. What Are Libraries? 🛠️**

A **library** is a collection of pre-written code that helps programmers complete tasks quickly. Instead of writing everything from scratch, libraries provide ready-made functions for handling data, creating graphs, and performing calculations.

### **Common Libraries for AI and Data Science**

📦 **Pandas** - Organizes and processes data (like tables in Excel).  
📊 **Seaborn** - Creates visually appealing charts.  
📉 **Matplotlib** - Makes customizable graphs and plots.  

Libraries save time and reduce errors by providing tested tools. 

💡 **Think:** How can these help in AI and data science? 🤔

---

## 🔍 **3. Hands-on: Exploring a Real Dataset**

### **[Open Google Colab](https://colab.research.google.com)**

1️⃣ Open your browser and go to **Google Colab**.\
2️⃣ Click **+ New notebook**.

✅ If everything went well, you should see a blank Colab notebook, ready for action!


---

##  💾 **4. Loading the Dataset**

### **➕🐍 Add a New Code Cell**
 
1️⃣ Click **+ Code** in the top left to add a new code cell.\
2️⃣ Copy and paste the following code into the new code cell:

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c441-7d40-8011-9a91-00cd12ce6b51)

```python
# Import the Pandas library to handle data
import pandas as pd  # Pandas helps us work with structured data like tables

# Create a simple dataset with missing values using a dictionary
# Each key (column name) has a list of values (rows of data)
data = pd.DataFrame({
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],  # Names of students
    'Age': [25, 30, 35, None, 40],  # Age column, with one missing value (None)
    'Score': [90, 85, 88, 92, None]  # Score column, with one missing value (None)
})

# Display the dataset as a table
data
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11423-42b4-8011-94af-bca7e583a7e4)

3️⃣ **Click Run (▶) and check the output!** 

✅ A small table should pop up with some missing values (NaN)🎉


🎯Challenge: Modify the dataset by adding at least two more students with their Name, Age, and Score values.<br>
💡Extra Tip: Edit the `data = pd.DataFrame({...})` section and include new students.
###
---

## 🧹 **5. Cleaning the Data**

###  **➕🐍 Add a New Code Cell**

1️⃣ Click **+ Code** to add another code cell.\
2️⃣ Copy and paste the following code:

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c5b1-eccc-8011-a4b3-1b72ada8e28d)

```python
# Fill missing values with the column average (only for numeric columns)
data.fillna(data.select_dtypes(include=['number']).mean(), inplace=True)  # Replace NaN with column averages

# Show the cleaned dataset
data  # Display the updated table
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d114b8-a10c-8011-90b6-d0d926d484ef)

3️⃣ **Click Run (▶) and check the output.** 

✅ No more NaN values! Everything should now have a meaningful number🎉


---

## 📊 **6. Analyzing the Data**

###  **➕🐍 Add a New Code Cell**

1️⃣ Click **+ Code** to add another code cell.\
2️⃣ Copy and paste the following code:

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c5fb-23f4-8011-bd28-c2866809e5a2)

```python
# Show basic statistics about the dataset
data.describe()  # This provides mean, min, max, and other useful statistics
```
🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11506-359c-8011-9bfa-6fbdef1b9a4f)

3️⃣ **Click Run (▶) to check the output.**


✅ Expect to see a table with average, min, max, and other useful stats🎉

---

## 📊 **7. Sorting Data to Find Top Scores**

###  **➕🐍 Add a New Code Cell**

1️⃣ Click **+ Code** to add another code cell.\
2️⃣ Copy and paste the following code:

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c63c-7de0-8011-94ab-763ff73678f8)

```python
# Sort students by their scores in descending order (highest first)
data.sort_values(by="Score", ascending=False)
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11558-66c0-8011-b65c-0a182f029c75)

3️⃣ **Click Run (▶) to check the output**

✅ The student with the highest score should now be at the top🎉

---

## 🎨 **8. Visualizing Data**

### 📊 **Bar Chart: Comparing Scores**

###  **Step 1: ➕🐍 Add a New Code Cell**

1️⃣ Click **+ Code** to add another code cell.\
2️⃣ Copy and paste the following code:

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c6ac-d528-8011-bbeb-cec6f5fcb318)

```python
# Import visualization libraries
import seaborn as sns  # Seaborn helps us make graphs
import matplotlib.pyplot as plt  # Matplotlib is used for chart customization

# Create a bar chart of student scores
sns.barplot(x=data["Name"], y=data["Score"])  # Draw a bar chart comparing scores
plt.title("Student Scores")  # Add a title to the chart
plt.show()  # Show the graph
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d11592-09a0-8011-91e1-e760be39df24)

3️⃣ **Click Run (▶) and check the output.**

✅ A neat bar chart should appear, making score comparisons much easier🎉

<img src="https://github.com/user-attachments/assets/a74306fc-e4ab-4238-b3d4-08c5069c4e89" width="350">

## 📊 **Scatter Plot: Finding Patterns**

###  **Step 2: ➕🐍 Add a New Code Cell**

4️⃣ Click **+ Code** to add another code cell.\
5️⃣ Copy and paste the following code:

🔗 [ChatGPT prompt to generate the code](https://chatgpt.com/share/67e4c6f7-8f18-8011-a647-af7b508c2a6d)

```python
# Import visualization libraries
import seaborn as sns  # Seaborn helps us make graphs
import matplotlib.pyplot as plt  # Matplotlib is used for chart customization

# Create a scatter plot for Age vs. Score
sns.scatterplot(x=data["Age"], y=data["Score"])  # Draw a scatter plot showing relationships
plt.title("Age vs. Score")  # Add a title to the chart
plt.show()  # Show the graph
```

🔗 [ChatGPT explanation for the code](https://chatgpt.com/share/67d115c6-76d8-8011-bf4a-a5c71eff8201)

6️⃣ **Click Run (▶) and check the output**

✅ A scatter plot should pop up, showing if older students tend to score higher🎉

<img src="https://github.com/user-attachments/assets/e11d1571-50bf-422a-a700-11286e630907" width="350">

---


## 🎯 9. Wrap-Up & Next Steps

🎉 Congratulations! You learned how to:

✅ Load a dataset 🔍📂<br>
✅ Clean missing values 🧹✨<br>
✅ Analyze data using basic statistics 📊<br>
✅ Create colorful visualizations 🎨<br>

🚀 Next Workshop: [🤖 Simple ML Model! ](https://github.com/DrAlzahrani/HPC-AI-Resources/wiki/personal-computer-simple-ml)

### 🔗 **Additional AI Resources** 📚   
- [Google Colab Guide](https://colab.research.google.com/)     
- [Python Introduction](https://www.w3schools.com/python/python_intro.asp) (Use only the two green buttons “Previous” and “Next” to navigate the tutorial and avoid ads.)<br>
- [W3Schools Data Science Introduction](https://www.w3schools.com/datascience/ds_introduction.asp)
- [Microsoft Foundations of data science for machine learning](https://learn.microsoft.com/en-us/training/paths/machine-learning-foundations-using-data-science/)

🎉 **You did it! Keep exploring AI, and see you at the next workshop!** 🚀

---

### 📝 Workshop Feedback Survey 

Thanks for completing this workshop!🎆

We'd love to hear what you think so we can make future workshops even better. 💡

📌 **[Survey link](https://docs.google.com/forms/d/e/1FAIpQLSfqnVP2EwGiwS1RLEvOUH8po0QTlQngSONuWELZ6d-YV5ulyg/viewform)**

---


