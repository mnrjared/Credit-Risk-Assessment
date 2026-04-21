import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import flask
import os

# Initialize
app = dash.Dash(__name__)
server = app.server

# FIX: This allows Dash to "see" the artifacts folder
@server.route('/artifacts/<path:path>')
def serve_artifacts(path):
    # This assumes artifacts is one level up from the src folder
    root_dir = os.path.dirname(os.getcwd())
    artifacts_path = os.path.join(root_dir, 'artifacts')
    return flask.send_from_directory(artifacts_path, path)

app.layout = html.Div(style={
    'backgroundColor': '#0f1115', 'color': '#f0f0f0', 'padding': '20px', 'fontFamily': 'Arial'
}, children=[
    html.H1("Credit Risk Intelligence System", style={'textAlign': 'center', 'color': '#00d4ff'}),
    
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}, children=[
        # Left Side: Inputs
        html.Div(style={
            'flex': '1', 'minWidth': '300px', 'backgroundColor': '#1a1d23', 'padding': '20px', 'borderRadius': '10px'
        }, children=[
            html.H3("Applicant Details", style={'color': '#FFD700'}),
            html.Label("Annual Income (R):"),
            dcc.Input(id='income', type='number', placeholder='e.g. 50000', style={'width': '100%', 'marginBottom': '10px'}),
            html.Label("Loan Amount (R):"),
            dcc.Input(id='loan', type='number', placeholder='e.g. 10000', style={'width': '100%', 'marginBottom': '10px'}),
            html.Label("Credit Score:"),
            dcc.Input(id='score', type='number', placeholder='e.g. 600', style={'width': '100%', 'marginBottom': '20px'}),
            html.Button('RUN ASSESSMENT', id='predict-btn', n_clicks=0, 
                        style={'width': '100%', 'padding': '10px', 'backgroundColor': '#FFD700', 'border': 'none', 'fontWeight': 'bold', 'color': '#121212'})
        ]),

        # Right Side: Results
        html.Div(style={'flex': '2', 'minWidth': '500px'}, children=[
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label='Risk Prediction', children=[
                    html.Div(id='prediction-output', style={'padding': '20px', 'textAlign': 'center'}),
                    dcc.Graph(id='importance-graph')
                ]),
                dcc.Tab(label='Model Evaluation (Artifacts)', children=[
                    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'padding': '20px'}, children=[
                        # We use the new /artifacts/ route here
                        html.Div([html.P("Confusion Matrix"), html.Img(src='/artifacts/confusion_matrix.png', width='100%')], style={'width': '45%', 'padding': '10px'}),
                        html.Div([html.P("ROC Curve"), html.Img(src='/artifacts/roc_curve.png', width='100%')], style={'width': '45%', 'padding': '10px'})
                    ])
                ])
            ])
        ])
    ])
])

@app.callback(
    [Output('prediction-output', 'children'),
     Output('importance-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('income', 'value'), State('loan', 'value'), State('score', 'value')]
)
def update_app(n_clicks, income, loan, score):
    # Adjust path because web_app.py is inside /src
    try:
        # Go up one level from /src to find /artifacts
        path = os.path.join(os.path.dirname(os.getcwd()), 'artifacts', 'feature_importance.csv')
        df_imp = pd.read_csv(path)
        fig = px.bar(df_imp, x='importance', y='feature', orientation='h', template='plotly_dark')
        fig.update_traces(marker_color='#FFD700') # Keep the yellow/gold theme
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    except:
        fig = px.bar(title="Artifact CSV not found in /artifacts folder", template='plotly_dark')

    if n_clicks == 0:
        return "", fig

    # Assessment Logic
    if score and score < 580:
        res, color = "Risk Level: HIGH", "#FF4C4C"
    else:
        res, color = "Risk Level: LOW", "#00FF7F"

    return html.H2(res, style={'color': color}), fig

if __name__ == '__main__':
    app.run(debug=True)