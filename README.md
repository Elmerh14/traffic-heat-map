
# 🚦 Colorado Crash Heatmap Dashboard

This project is an interactive dashboard for visualizing crash data in Colorado, built with Python, Dash, Plotly, and Bootstrap.

---

## ✅ Prerequisites

- Python 3.8 or later
- Git (optional, if cloning the repository)
- A terminal or command prompt
- Internet connection (to install packages)

---

## 🔧 Setup Instructions

### 1. Create a Virtual Environment

#### 🐧 On Unix/Linux/macOS:
```bash
python3 -m venv env
source env/bin/activate
```

#### 🪟 On Windows:
```bash
python -m venv env
env\Scripts\activate
```

---

### 2. Install Dependencies

Make sure you're in the activated virtual environment, then run:

```bash
pip install -r requirements.txt
```

---

### 3. Run the Dashboard

```bash
python3 app.py
```

Once the server is running, open your browser and go to:

```
http://127.0.0.1:8050/
```



## 🛑 Deactivating the Environment (when done)

```bash
deactivate
```

---

## 📂 File Highlights

- `app.py` – main dashboard UI
- `requirements.txt` – pinned package list
- `crash_risk_heatmap.html` / `crash_clusters_map.html` – heatmap visualizations to embed

---

## 🧠 Notes

- Ensure you have `crash_risk_heatmap.html` and `crash_clusters_map.html` in the root folder before running.
- If you regenerate heatmaps, make sure they save with the same names or update `app.py`.

---

##  AI

- There was the use of GPT for the code to use stuff like Pytorch, Pandas, Transformers, Joblib, folium to put the data together. The data was found from the crash reports here in Colorado in 2024 
    - https://www.codot.gov/safety/traffic-safety/data-analysis/crash-data
- Model that was used to train was the distilBERT
- DistilBERT tokenizer knows how to break down English sentences which was from the Huggin Face model
- In the heat map only 2000 data rows were used since more rows was taking a good amount of time to train and then create the heat map based on the risk the model gave it
- There is still a big gap for us to understand the model distilBERT and how it is doing the predicting since this is our first time doing something like this
- There was a lot of things that were learned in the process about like how the model is getting the data and how it processes it, but there is still more to explore to fully understand 

---

## Data

- Columns from the data set that were used was:
    - CUID
    - Crash Type
    - Road conditions
    - Weather Conditions
    - Lighting Conditions
    - Crash Date
    - Crash Time
    - City
    - County
    - TU-1 Driver Action
    - TU-2 Driver Action
    - TU-1 Human Contributing Factor
    - TU-2 Human Contributing Factor
    - TU-1 Age
    - TU-2 Age
    - Latitude
    - Longitude

--

Feel free to customize the project structure and add more features as needed!
