import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import flask
import os
import joblib # New import for loading .pkl models

# 1. INITIALIZE & SERVER ROUTE
app = dash.Dash(__name__)
server = app.server

# special route to serve images from /artifacts
@server.route('/artifacts/<path:path>')
def serve_artifacts(path):
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    artifacts_path = os.path.join(root_dir, 'artifacts')
    return flask.send_from_directory(artifacts_path, path)

# LOAD MODELS
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
def load_model(name):
    path = os.path.join(root_dir, 'artifacts', name)
    try:
        return joblib.load(path)
    except:
        return None

model1 = load_model('model_1.pkl')
model2 = load_model('model_2.pkl')

# 2. STYLING & LAYOUT
app.layout = html.Div(style={
    'backgroundColor': '#0b0e14', 'color': '#e0e0e0', 'padding': '40px', 
    'minHeight': '100vh', 'fontFamily': 'Segoe UI, sans-serif'
}, children=[
    html.Div([
        html.P("BC ANALYTICS", style={'color': '#00d4ff', 'fontSize': '12px', 'fontWeight': 'bold'}),
        html.H1("Credit Risk Segmentation & Decision Support", style={'marginTop': '0px', 'fontSize': '28px'}),
        html.P("Explore model insights, then enter details for an ML-driven risk assessment.", style={'color': '#888'})
    ], style={'marginBottom': '30px'}),

    html.Div(style={'display': 'flex', 'gap': '20px', 'flexWrap': 'wrap'}, children=[
        
        # Left Column: Model Insights
        html.Div(style={
            'flex': '1', 'backgroundColor': '#161b22', 'padding': '25px', 
            'borderRadius': '12px', 'minWidth': '450px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.3)'
        }, children=[
            html.H3("Model Insights (Artifacts)", style={'fontSize': '18px', 'marginBottom': '20px'}),
            
            # Row for confusion matrix and ROC
            html.Div(style={'display': 'flex', 'gap': '10px', 'marginBottom': '20px'}, children=[
                html.Img(src='/artifacts/confusion_matrix.png', style={'width': '48%', 'borderRadius': '5px'}),
                html.Img(src='/artifacts/roc_curve.png', style={'width': '48%', 'borderRadius': '5px'}),
            ]),
            
            # Changed from dcc.Graph to static png
            html.P("Feature Importance — mean Impact", style={'color': '#888', 'fontSize': '14px'}),
            html.Img(src='/artifacts/feature_importance.png', style={'width': '100%', 'borderRadius': '5px', 'marginTop': '10px'}),
        ]),

        # Right Column: Input Section
        html.Div(style={
            'flex': '1', 'backgroundColor': '#161b22', 'padding': '25px', 
            'borderRadius': '12px', 'minWidth': '450px', 'boxShadow': '0 4px 15px rgba(0,0,0,0.3)'
        }, children=[
            html.H3("Applicant Input", style={'fontSize': '18px', 'marginBottom': '20px'}),
            
            html.Div(style={'display': 'grid', 'gridTemplateColumns': '1fr 1fr', 'gap': '15px'}, children=[
                html.Div([html.Label("Annual Income (R)"), dcc.Input(id='income', type='number', className='custom-input', value=0)]),
                html.Div([html.Label("Loan Amount (R)"), dcc.Input(id='loan', type='number', className='custom-input', value=0)]),
                html.Div([html.Label("Credit Score"), dcc.Input(id='score', type='number', className='custom-input', value=0)]),
                html.Div([html.Label("Employment Length (Years)"), dcc.Input(id='emp', type='number', className='custom-input', value=0)]),
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
    Output('prediction-output', 'children'),
    Input('predict-btn', 'n_clicks'),
    [State('income', 'value'), State('loan', 'value'), State('score', 'value'), State('emp', 'value')]
)
def update_app(n_clicks, income, loan, score, emp):
    if n_clicks == 0:
        return ""

    if model1 is None or model2 is None:
        return html.H4("Missing one or both .pkl models in /artifacts", style={'color': 'orange'})

    # Prepare input for the model. 
    # NOTE: The order must match exactly what P2 used to train the model.
    input_data = pd.DataFrame([[income, loan, score, emp]], 
                              columns=['income', 'loan', 'score', 'emp'])
    
    try:
        p1 = model1.predict(input_data)[0]
        p2 = model2.predict(input_data)[0]

    # Result logic
        if p1 == 1 and p2 == 1:
            res, color, msg = "HIGH RISK", "#ff4c4c", "Both models indicate high risk."
        elif p1 == 0 and p2 == 0:
            res, color, msg = "LOW RISK", "#00ff7f", "Both models indicate low risk."
        else:
            res, color, msg = "MODERATE RISK", "yellow", "Models disagree (Ambiguous case)."

        return html.Div([
            html.H2(res, style={'color': color, 'margin': '0'}),
            html.P(msg, style={'color': '#888', 'marginTop': '5px'})
        ])
            
    except Exception as e:
        return html.P(f"Feature Error: {str(e)}", style={'color': 'yellow'})

if __name__ == '__main__':
    app.run(debug=True)