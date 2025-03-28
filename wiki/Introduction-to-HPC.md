# 🚀 Unlocking the Future: How HPC and AI Power Research Across Disciplines

**Objective:** Learn how High-Performance Computing (HPC) and Artificial Intelligence (AI) can accelerate research in any field, from biology to history. This workshop is designed for **beginners** with no prior knowledge of AI, HPC, or Python.

---

## 🧠 What You’ll Learn

- What HPC and AI are and why they matter.
- How to use Python and Jupyter notebooks for data analysis.
- Real-world examples of HPC and AI in action.
- How to run code on your **local computer** and on **HPC with GPU**.

---

## 🛠️ Part 1: Using Your Local Computer

### Example 1: Analyzing Book Genres

Let’s analyze a dataset of book genres and visualize the most popular genres using Python.

```python
# Step 1: Import the necessary libraries
import pandas as pd  # Pandas is used for working with datasets (like Excel for Python)
import matplotlib.pyplot as plt  # Matplotlib is used for creating graphs and charts

# Step 2: Load the dataset of book genres
data = pd.read_csv("goodreads_books.csv")  # This reads the dataset from a file called "goodreads_books.csv"

# Step 3: Count how many books are in each genre
genre_counts = data['Genre'].value_counts()  # This counts the number of books in each genre

# Step 4: Create a bar chart to visualize the most popular genres
genre_counts.plot(kind='bar', color='skyblue', edgecolor='black')  # This creates a bar chart
plt.title("Most Popular Book Genres")  # This adds a title to the chart
plt.xlabel("Genre")  # This labels the x-axis
plt.ylabel("Number of Books")  # This labels the y-axis
plt.show()  # This displays the chart

### Example 1: Analyzing Book Genres

Let’s analyze a dataset of book genres and visualize the most popular genres using Python.