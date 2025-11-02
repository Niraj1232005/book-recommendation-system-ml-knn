# 📚 Book Recommendation System

<div align="center">

![Python](https://img.shields.io


![Streamlit](https://img.shields.io/badge/Streamlit-1.28shields.io/badge/scikit--learn-1






**A powerful ML-powered book recommendation system with an intuitive Streamlit interface**

[Features](#-features) -  [Demo](#-demo) -  [Installation](#-installation) -  [Usage](#-usage) -  [Tech Stack](#-tech-stack)

</div>

***

## 🌟 Features

<table>
<tr>
<td width="50%">

### 🔍 **Smart Search**
- Fuzzy text matching using RapidFuzz
- TF-IDF vectorization for semantic search
- Search by book title or author name
- Typo-tolerant query handling

</td>
<td width="50%">

### 🎯 **Personalized Recommendations**
- k-Nearest Neighbors (KNN) algorithm
- Content-based filtering
- Rating-aware suggestions
- Similar books based on features

</td>
</tr>
<tr>
<td width="50%">

### 🏆 **Top Rankings**
- Most-rated books
- Top authors by book count
- Highest-rated titles
- Popular book analytics

</td>
<td width="50%">

### 📊 **Data Visualization**
- Interactive charts with Matplotlib & Seaborn
- Rating distribution analysis
- Correlation heatmaps
- Publication trends

</td>
</tr>
</table>

***

## 🎬 Demo

```bash
# Quick start - Get the app running in 3 commands
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system
pip install -r requirements.txt && streamlit run app.py
```

***

## 📂 Project Structure

```
book-recommendation-system/
│
├── 📄 app.py                    # Main Streamlit application
├── 📊 books.csv                 # Dataset (book metadata)
├── 📋 requirements.txt          # Python dependencies
├── 📖 README.md                 # Project documentation
├── 🚫 .gitignore               # Git ignore file
└── 📝 app.log                   # Application logs (generated)
```

***

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (optional, for cloning)

### Step-by-Step Setup

#### 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/your-username/book-recommendation-system.git
cd book-recommendation-system
```

#### 2️⃣ **Create Virtual Environment** (Recommended)

<details>
<summary><b>Windows</b></summary>

```bash
python -m venv venv
venv\Scripts\activate
```
</details>

<details>
<summary><b>macOS/Linux</b></summary>

```bash
python3 -m venv venv
source venv/bin/activate
```
</details>

#### 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

#### 4️⃣ **Run the Application**

```bash
streamlit run app.py
```

#### 5️⃣ **Open in Browser**

The app will automatically open at:
```
🌐 Local URL: http://localhost:8501
```

***

## 💻 Usage

### Search for Books

```python
# In the app interface
1. Navigate to "Search Books" section
2. Enter a book title (e.g., "Harry Potter")
3. View fuzzy-matched results with ratings
```

### Get Recommendations

```python
# Find similar books
1. Go to "Book Recommendations" tab
2. Enter a book you like
3. Receive 10 personalized suggestions
```

### Explore Top Books

```python
# Discover popular titles
1. Select "Top Books" from sidebar
2. View most-rated books
3. Explore top authors
```

### Visualize Data

```python
# Analyze book trends
1. Open "Data Insights" section
2. View interactive charts
3. Explore rating distributions
```

***

## 🧠 Tech Stack

<table>
<tr>
<td align="center" width="20%">
<img src="https://streamlit.io/images/brand/streamlit-mark-color.png" width="50px" /><br />
<b>Streamlit</b><br />
Web Framework
</td>
<td align="center" width="20%">
<img src="https://upload.wikimedia.org/wikipedia/commons/e/ed/Pandas_logo.svg" width="50px" /><br />
<b>Pandas</b><br />
Data Processing
</td>
<td align="center" width="20%">
<img src="https://upload.wikimedia.org/wikipedia/commons/0/05/Scikit_learn_logo_small.svg" width="50px" /><br />
<b>Scikit-learn</b><br />
ML Algorithms
</td>
<td align="center" width="20%">
<img src="https://matplotlib.org/stable/_static/logo2.svg" width="50px" /><br />
<b>Matplotlib</b><br />
Visualization
</td>
<td align="center" width="20%">
<img src="https://seaborn.pydata.org/_static/logo-wide-lightbg.svg" width="50px" /><br />
<b>Seaborn</b><br />
Statistical Plots
</td>
</tr>
</table>

### Core Technologies

| Technology | Purpose | Version |
|------------|---------|---------|
| **Python** | Core programming language | 3.8+ |
| **Streamlit** | Web app framework | 1.28+ |
| **Pandas** | Data manipulation & analysis | 2.0+ |
| **NumPy** | Numerical computing | 1.24+ |
| **Scikit-learn** | Machine learning (KNN, TF-IDF) | 1.3+ |
| **RapidFuzz** | Fuzzy string matching | 3.0+ |
| **Matplotlib** | Data visualization | 3.7+ |
| **Seaborn** | Statistical visualization | 0.12+ |

***

## 🤖 Machine Learning Pipeline

### 1. Data Preprocessing

```python
✓ Remove duplicate entries
✓ Handle missing values (fillna/dropna)
✓ Normalize text fields (lowercase, strip)
✓ Encode categorical variables
```

### 2. Feature Engineering

```python
Features Used:
├── average_rating          # Book's average rating
├── ratings_count          # Total number of ratings
├── language_code          # Encoded language
├── num_pages             # Number of pages
└── rating_bins           # Discretized rating categories
```

### 3. Model Architecture

```python
Algorithm: k-Nearest Neighbors (KNN)
├── Distance Metric: Euclidean
├── n_neighbors: 10
├── Algorithm: auto
└── Weights: uniform
```

### 4. Search Enhancement

```python
Hybrid Search System:
├── TF-IDF Vectorization (Semantic Similarity)
├── RapidFuzz (Fuzzy String Matching)
└── Combined Scoring (Weighted Average)
```

***

## 📊 Dataset

The system uses a comprehensive book dataset (`books.csv`) containing:

- **Book Titles**: 10,000+ unique books
- **Authors**: Multiple authors per book
- **Ratings**: Average ratings and counts
- **Metadata**: ISBN, publisher, publication year, language
- **Pages**: Book length information

***

## ⚡ Performance Optimizations

```python
@st.cache_data    # Cache data loading and preprocessing
@st.cache_resource # Cache ML model training
```

- **Lazy Loading**: Data loaded only when needed
- **Streamlit Caching**: Prevents redundant computations
- **Efficient Search**: Optimized fuzzy matching algorithms
- **Logging**: Track performance bottlenecks

***

## 🔮 Future Enhancements

- [ ] **User Authentication**: Personal reading lists and history
- [ ] **Genre-Based Filtering**: Recommendations by category
- [ ] **Collaborative Filtering**: User-based recommendations
- [ ] **Book Cover Images**: Visual book browsing
- [ ] **Export Features**: Save recommendations as PDF/CSV
- [ ] **API Integration**: Real-time book data from Google Books API
- [ ] **Deployment**: Host on Streamlit Cloud/Heroku/AWS

***

## 🐛 Troubleshooting

<details>
<summary><b>Import Error: No module named 'streamlit'</b></summary>

```bash
# Solution: Install dependencies
pip install -r requirements.txt
```
</details>

<details>
<summary><b>Port Already in Use</b></summary>

```bash
# Solution: Run on different port
streamlit run app.py --server.port 8502
```
</details>

<details>
<summary><b>Dataset Not Found</b></summary>

```bash
# Solution: Ensure books.csv is in the same directory as app.py
ls -la books.csv
```
</details>

***

## 📝 Requirements.txt

```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
rapidfuzz>=3.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
```

***

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add some AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

***

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Your Name

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software...
```

***

## 👨‍💻 Author

<div align="center">

### **Your Name**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor//img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor.shields.io/badge/Email-D14836?style=for-the-badgeolio](https://img.shields.io/badge/Portfolio-000000?style=for-the-badge&

</div>

***

## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=your-username/book-recommendation-system/#your-username/book-recommendation-

## 📞 Support

Having issues? Contact me:

- 📧 **Email**: your@email.com
- 💬 **Discord**: YourDiscord#1234
- 🐦 **Twitter**: [@yourusername](https://twitter.com/yourusername)

***

<div align="center">

**Made with ❤️ and Python**

⭐ **Star this repo if you find it helpful!** ⭐

</div>
