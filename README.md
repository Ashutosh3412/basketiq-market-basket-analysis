# 🛒 BasketIQ — Market Basket Analysis

**An interactive retail analytics dashboard powered by the Apriori algorithm.**

BasketIQ is a Python-based data mining application that discovers frequently purchased products and identifies relationships between items in transaction data. It provides interactive visualizations, configurable analysis parameters, and downloadable results through a modern Streamlit dashboard.

## ✨ Features

- **Interactive Dashboard:** View transaction statistics, product patterns, and association-rule insights.
- **Frequent Itemset Mining:** Discover products and product combinations that occur frequently.
- **Association Rule Generation:** Identify relationships between products using the Apriori algorithm.
- **Configurable Parameters:** Adjust minimum support, minimum confidence, maximum itemset size, and the number of transactions analyzed.
- **Interactive Visualizations:** Explore product combinations through charts with hover details and visual highlighting.
- **Rule Analysis:** Evaluate associations using support, confidence, and lift.
- **Dataset Explorer:** Inspect original transaction data, cleaned records, and the transaction basket matrix.
- **CSV Export:** Download frequent itemsets and association rules for further analysis.
- **Animated Interface:** Use a dark-themed dashboard with visual transitions and interactive elements.

## 🎯 Project Objective

The objective of BasketIQ is to demonstrate how association rule mining can be used to discover patterns in retail transaction data.

Retailers can use these patterns to explore product relationships, plan promotions, improve product placement, and support cross-selling strategies.

## 🧠 What Is the Apriori Algorithm?

Apriori is an association rule mining algorithm used to find frequent itemsets in transactional datasets.

It uses the principle that if an itemset is frequent, all its non-empty subsets must also be frequent. This helps eliminate candidate combinations that cannot meet the chosen support threshold.

BasketIQ uses Apriori to identify frequent product combinations and then generates association rules to measure relationships between products.

### Key Metrics

| Metric | Meaning |
|---|---|
| **Support** | The proportion of transactions containing an itemset. |
| **Confidence** | The proportion of transactions containing the antecedent that also contain the consequent. |
| **Lift** | Measures how frequently the consequent occurs with the antecedent compared with its overall frequency. |

**Interpretation of lift:**
- Lift greater than 1 indicates a positive association.
- Lift equal to 1 indicates statistical independence under the measured frequencies.
- Lift below 1 indicates a negative association.

A high lift value does not automatically mean a rule is reliable or commercially useful. Support, confidence, sample size, and the dataset should also be considered.

## ⚙️ Technology Stack

- **Python** — Core programming language
- **Streamlit** — Web application interface
- **Pandas** — Data preparation and manipulation
- **Mlxtend** — Apriori algorithm and association rule generation
- **Plotly** — Interactive data visualizations
- **OpenPyXL** — Excel file processing

## 📁 Project Structure

```text
basketiq-market-basket-analysis/
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

The application accepts transaction data uploaded by the user. Keep large datasets and the Python virtual environment out of the Git repository unless there is a specific reason to include them.

## 🚀 Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/basketiq-market-basket-analysis.git
cd basketiq-market-basket-analysis
```

Replace `YOUR_USERNAME` with your GitHub username.

### 2. Create a virtual environment

On Windows:

```powershell
python -m venv .venv
```

### 3. Install dependencies

On Windows, you can run the environment's Python directly without activating it:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

### 4. Start the application

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py
```

Open the local URL displayed in the terminal, usually:

```text
http://localhost:8501
```

## 📊 How to Use BasketIQ

1. Open the application.
2. Read the project overview and methodology.
3. Upload a transaction dataset in CSV or Excel format.
4. Configure the minimum support and confidence thresholds.
5. Choose the maximum number of products per itemset and the transaction sample size.
6. Explore frequent itemsets and association rules.
7. Inspect the dataset and review the visualizations.
8. Download the results as CSV files for further analysis.

## 🗂️ Dataset Format

The uploaded dataset must contain at least these columns:

| Column | Description |
|---|---|
| `InvoiceNo` | Transaction or invoice identifier |
| `Description` | Product name or description |
| `Quantity` | Quantity purchased |

Each invoice represents a transaction, and each product description identifies an item within that transaction.

The application preprocesses records, removes cancelled invoices and invalid purchase quantities, and creates a basket matrix indicating whether each product appears in each transaction.

For reproducible analysis, use a consistent dataset and record the thresholds and transaction sample size used.

## 📈 Applications

Market basket analysis can help businesses:

- Identify products that are frequently purchased together.
- Explore cross-selling opportunities.
- Design product bundles and promotions.
- Support retail merchandising decisions.
- Understand purchasing patterns in transaction data.

The discovered associations show patterns in the observed data; they do not establish that one purchase causes another.

## ⚠️ Limitations

- The quality of the output depends on the dataset and selected thresholds.
- Very high lift values can occur for rare item combinations, so support and transaction counts should be considered.
- Analyzing only a subset of transactions may not represent the complete dataset.
- Results describe observed associations rather than guaranteeing future purchasing behavior.
- Performance can vary with dataset size, itemset length, and analysis parameters.

## 🎓 Academic Project

BasketIQ was developed as a Data Mining / Business Analytics implementation project to demonstrate frequent itemset discovery and association rule mining using the Apriori algorithm.

**Core objective:** Convert transaction data into interpretable purchasing patterns through data preprocessing, frequent itemset mining, rule evaluation, and visualization.

## 📜 License

This project is intended for educational and demonstration purposes. A license has not been specified yet. Add a suitable open-source license if you decide to distribute or reuse the project publicly.
