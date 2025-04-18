import folium
from folium.plugins import MarkerCluster
import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# Load crash data
df = pd.read_excel('CDOTRM_CD_Crash_Listing_-_2024.xlsx')
df.columns = df.columns.str.strip()

# Drop rows with missing coordinates
df = df.dropna(subset=['Latitude', 'Longitude'])

# Convert to NumPy array of coordinate pairs
coords = df[['Latitude', 'Longitude']].to_numpy()

# Convert miles to radians (for Haversine metric)
kms_per_radian = 6371.0088
epsilon = 0.4 / kms_per_radian  # ~0.25 miles

# Apply DBSCAN clustering
db = DBSCAN(eps=epsilon, min_samples=15, algorithm='ball_tree', metric='haversine').fit(np.radians(coords))
df['Cluster'] = db.labels_

# Filter out noise (cluster = -1)
df = df[df['Cluster'] != -1]

# Create base map centered on crash data
m = folium.Map(location=[df['Latitude'].mean(), df['Longitude'].mean()], zoom_start=9, tiles='cartodbpositron')

# Add marker clustering
marker_cluster = MarkerCluster().add_to(m)

# Define color cycle for cluster ID visualization
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightgray', 'pink', 'darkblue', 'darkgreen']

# Plot each crash point
for _, row in df.iterrows():
    cluster_id = row['Cluster']
    color = colors[cluster_id % len(colors)]
    folium.CircleMarker(
        location=(row['Latitude'], row['Longitude']),
        radius=2,
        color=color,
        fill=True,
        fill_opacity=0.7
    ).add_to(marker_cluster)

# Save the map
m.save("crash_clusters_map.html")
print("Map saved as crash_clusters_map.html. Open it in your browser to explore!")
