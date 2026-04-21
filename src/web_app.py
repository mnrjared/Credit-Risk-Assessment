import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import flask
import os

# 1. INITIALIZE & SERVER ROUTE
app = dash.Dash(__name__)
server = app.server

# This special route allows the browser to see the images in /artifacts
@server.route('/artifacts/<path:path>')
def serve_artifacts(path):
    root_dir = os.path.dirname(os.getcwd()) # Goes up from /src to project root
    artifacts_path = os.path.join(root_dir, 'artifacts')
    return flask.send_from_directory(artifacts_path, path)

# 2. STYLING & LAYOUT
app.layout = html.Div(style={
    'backgroundColor': '#0b0e14', 'color': '#e0e0e0', 'padding': '40px', 
    'minHeight': '100vh', 'fontFamily': 'Segoe UI, sans-serif'
}, children=[
    # Header
    html.Div([
        html.P("BC ANALYTICS", style={'color': '#00d4ff', 'fontSize': '12px', 'fontWeight': 'bold'}),
        html.H1("Credit Risk Segmentation & Decision Support", style={'marginTop': '0px', 'fontSize': '28px'}),
        html.P("Explore model insights, then enter details to get a predicted risk assessment.", style={'color': '#888'})
    ], style={'marginBottom': '30px'}),

    # Main Content Container (Flexbox for the two columns)
    html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
        
        # Left Column: Model Insights (The Card)
        html.Div(style={
            'flex': '1', 'backgroundColor': '#161b22', 'padding': '25px', 
            'borderRadius': '12px', 'minWidth': '450px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.3)'
        }, children=[
            html.H3("Model Insights (from artifacts)", style={'fontSize': '18px', 'marginBottom': '20px'}),
            
            # Row for small images
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '20px'}, children=[
                html.Img(src='/artifacts/confusion_matrix.png', style={'width': '48%', 'borderRadius': '5px'}),
                html.Img(src='/artifacts/roc_curve.png', style={'width': '48%', 'borderRadius': '5px'}),
            ]),
            
            # Feature Importance Graph (The Interactive Part)
            html.P("Top Features — mean Impact", style={'color': '#888', 'fontSize': '14px'}),
            dcc.Graph(id='importance-graph', config={'displayModeBar': False}, style={'height': '400px'})
        ]),

        # Right Column: Input Section (The Card)
        html.Div(style={
            'flex': '1', 'backgroundColor': '#161b22', 'padding': '25px', 
            'borderRadius': '12px', 'minWidth': '450px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.3)'
        }, children=[
            html.H3("Applicant Input", style={'fontSize': '18px', 'marginBottom': '20px'}),
            
            # Grid for inputs
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'}, children=[
                html.Div([html.Label("Annual Income (R)"), dcc.Input(id='income', type='number', className='custom-input')]),
                html.Div([html.Label("Loan Amount (R)"), dcc.Input(id='loan', type='number', className='custom-input')]),
                html.Div([html.Label("Credit Score"), dcc.Input(id='score', type='number', className='custom-input')]),
                html.Div([html.Label("Employment Length"), dcc.Input(id='emp', type='number', className='custom-input')]),
            ]),
            
            html.Button('Run assessment', id='predict-btn', n_clicks=0, style={
                'width': '100%', 'marginTop': '30px', 'padding': '12px', 
                'backgroundColor': '#00d4ff', 'color': '#0b0e14', 'border': 'none', 
                'borderRadius': '8px', 'fontWeight': 'bold', 'cursor': 'pointer'
            }),

            html.Div(id='prediction-output', style={'marginTop': '25px', 'textAlign': 'center', 'fontSize': '20px'})
        ])
    ])
])

# 3. LOGIC & CALLBACK
@app.callback(
    [Output('prediction-output', 'children'),
     Output('importance-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('income', 'value'), State('loan', 'value'), State('score', 'value')]
)
def update_app(n_clicks, income, loan, score):
    # Path logic: Go up from /src to find /artifacts/feature_importance.csv 
    try:
        path = os.path.join(os.path.dirname(os.getcwd()), 'artifacts', 'feature_importance.csv')
        df_imp = pd.read_csv(path)
        fig = px.bar(df_imp, x='importance', y='feature', orientation='h', template='plotly_dark')
        fig.update_traces(marker_color='#00d4ff')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=0, b=0))
    except:
        fig = px.bar(title="Artifact CSV not found", template='plotly_dark')

    if n_clicks == 0:
        return "", fig

    # Logic
    res = "HIGH RISK" if score and score < 600 else "LOW RISK"
    color = "#ff4c4c" if "HIGH" in res else "#00ff7f"
    
    return html.H2(res, style={'color': color}), fig

if __name__ == '__main__':
    app.run(debug=True)