import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import os

# 1. INITIALIZE
app = dash.Dash(__name__)
server = app.server

# 2. LOAD DATASET FOR ANALYSIS
# We use the dataset to show the user where they sit compared to the data
try:
    df_full = pd.read_csv('data/credit_risk_dataset.csv')
    avg_income = df_full['person_income'].mean()
except Exception as e:
    print(f"Error loading dataset: {e}")
    avg_income = 50000

# 3. LAYOUT
app.layout = html.Div(style={
    'backgroundColor': '#0f1115', 'color': '#f0f0f0', 'padding': '20px', 'fontFamily': 'Arial'
}, children=[
    html.H1("Credit Risk Intelligence System", style={'textAlign': 'center', 'color': '#00d4ff'}),
    
    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'gap': '20px'}, children=[
        # Input Section
        html.Div(style={
            'flex': '1', 'minWidth': '300px', 'backgroundColor': '#1a1d23', 'padding': '20px', 'borderRadius': '10px'
        }, children=[
            html.H3("Applicant Details"),
            html.Label("Annual Income (R):"),
            dcc.Input(id='income', type='number', value=avg_income, style={'width': '100%', 'marginBottom': '10px'}),
            
            html.Label("Loan Amount (R):"),
            dcc.Input(id='loan', type='number', placeholder='e.g. 50000', style={'width': '100%', 'marginBottom': '10px'}),
            
            html.Label("Credit Score:"),
            dcc.Input(id='score', type='number', placeholder='e.g. 600', style={'width': '100%', 'marginBottom': '20px'}),
            
            html.Button('RUN ASSESSMENT', id='predict-btn', n_clicks=0, 
                        style={'width': '100%', 'padding': '10px', 'backgroundColor': '#00d4ff', 'border': 'none', 'fontWeight': 'bold'})
        ]),

        # Results and Artifacts Section
        html.Div(style={'flex': '2', 'minWidth': '500px'}, children=[
            dcc.Tabs(id="tabs", children=[
                dcc.Tab(label='Risk Prediction', style={'backgroundColor': '#1a1d23'}, children=[
                    html.Div(id='prediction-output', style={'padding': '20px', 'textAlign': 'center'}),
                    dcc.Graph(id='importance-graph')
                ]),
                dcc.Tab(label='Model Evaluation (Artifacts)', style={'backgroundColor': '#1a1d23'}, children=[
                    html.Div(style={'display': 'flex', 'flexWrap': 'wrap', 'justifyContent': 'center', 'padding': '20px'}, children=[
                        # Images must be in the /assets folder to show up like this
                        html.Div([html.P("Confusion Matrix"), html.Img(src='/assets/confusion_matrix.png', width='90%')], style={'width': '45%'}),
                        html.Div([html.P("ROC Curve"), html.Img(src='/assets/roc_curve.png', width='90%')], style={'width': '45%'})
                    ])
                ])
            ])
        ])
    ])
])

# 4. CALLBACK
@app.callback(
    [Output('prediction-output', 'children'),
     Output('importance-graph', 'figure')],
    Input('predict-btn', 'n_clicks'),
    [State('income', 'value'), State('loan', 'value'), State('score', 'value')]
)
def process_assessment(n_clicks, income, loan, score):
    # Load Feature Importance from P2
    try:
        df_imp = pd.read_csv('artifacts/feature_importance.csv')
        fig = px.bar(df_imp, x='importance', y='feature', orientation='h', template='plotly_dark')
        fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
    except:
        fig = px.bar(title="Loading Feature Data...")

    if n_clicks == 0:
        return "Enter data and click Assess", fig

    # Logic using Dataset Averages
    # Simple risk logic for demonstration
    if score and score < 580:
        res = "Risk Level: HIGH"
        color = "red"
    elif income and loan and (loan > income * 0.5):
        res = "Risk Level: MODERATE (Loan exceeds 50% of Income)"
        color = "orange"
    else:
        res = "Risk Level: LOW (Likely Approved)"
        color = "green"

    return html.H2(res, style={'color': color}), fig

if __name__ == '__main__':
    app.run(debug=True)