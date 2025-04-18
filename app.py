import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output
import os

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.QUARTZ] 
)

# === Layout ===
app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Colorado Crash Heatmap Viewer"), className="text-center mb-2 mt-2")
    ]),

    dbc.Row([
        dbc.Col([
            dbc.ButtonGroup(
                [
                    dbc.Button("Crash Clusters", id="btn-clusters", n_clicks=0, color="secondary",  ),
                    dbc.Button("Crash Risk Heatmap", id="btn-risk", n_clicks=0, color="primary")
                ],
                size="lg",
                className="d-flex justify-content-center"
            )

        ], width=4)
    ],justify="center"),

    dbc.Row([
        dbc.Col(
            html.Iframe(
                id='map-frame',
                srcDoc=open("crash_risk_heatmap.html", "r").read(),
                width="100%",
                height="700px",
                style={
                    "border": "none",
                    "marginTop": "20px",
                    "borderRadius": "15px", 
                    "overflow": "hidden"
                }
            ), width=12
        )
    ])
], fluid=True)

# === Callback to switch maps ===
@app.callback(
    Output('map-frame', 'srcDoc'),
    [Input('btn-clusters', 'n_clicks'), Input('btn-risk', 'n_clicks')]
)
def switch_map(n1, n2):
    if n2 > n1:
        filename = "crash_risk_heatmap.html"
    else:
        filename = "crash_clusters_map.html"
    with open(filename, "r") as f:
        return f.read()

# === Run app ===
if __name__ == '__main__':
    app.run(debug=True)
