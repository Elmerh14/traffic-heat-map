
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

Feel free to customize the project structure and add more features as needed!
