# debug_spacex_dash.py
import os, sys
import pandas as pd
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px

DATA_FILE = "spacex_launch_dash.csv"
REMOTE = ("https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/"
          "IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv")

# Load data with fallback and print diagnostics
def load_df():
    try:
        if os.path.exists(DATA_FILE):
            df = pd.read_csv(DATA_FILE)
            src = DATA_FILE
        else:
            df = pd.read_csv(REMOTE)
            src = REMOTE
        df.columns = [c.strip() for c in df.columns]
        print(f"[INFO] Loaded from {src}", file=sys.stderr)
        print(f"[INFO] shape: {df.shape}", file=sys.stderr)
        print(f"[INFO] columns: {df.columns.tolist()}", file=sys.stderr)
        return df
    except Exception as e:
        print("[ERROR] reading dataset:", e, file=sys.stderr)
        # empty df with common columns
        cols = ['Launch Site','Payload Mass (kg)','class','Booster Version Category','Flight Number']
        return pd.DataFrame(columns=cols)

df = load_df()

# Try to normalize columns to the lab names
# ensure 'class' column exists (lowercase)
if 'class' not in df.columns and 'Class' in df.columns:
    df['class'] = df['Class']

# find payload column candidates and normalize to 'Payload Mass (kg)'
payload_candidates = ['Payload Mass (kg)','PayloadMass','Payload Mass','PayloadMass(kg)']
for c in payload_candidates:
    if c in df.columns:
        df = df.rename(columns={c: 'Payload Mass (kg)'})
        break

# ensure Launch Site present with some variant
launch_candidates = ['Launch Site','LaunchSite','Launch_Site','Site']
for c in launch_candidates:
    if c in df.columns and 'Launch Site' not in df.columns:
        df = df.rename(columns={c: 'Launch Site'})
        break

# ensure booster column exists
if 'Booster Version Category' not in df.columns and 'Booster Version' in df.columns:
    df['Booster Version Category'] = df['Booster Version']

# safe min/max for slider
try:
    min_payload = int(pd.to_numeric(df['Payload Mass (kg)'], errors='coerce').min())
    max_payload = int(pd.to_numeric(df['Payload Mass (kg)'], errors='coerce').max())
except:
    min_payload, max_payload = 0, 10000

# build dash
app = dash.Dash(__name__)
app.title = "SpaceX Debug Dashboard"

# dropdown options (include ALL)
sites = sorted(pd.Series(df.get('Launch Site', pd.Series([], dtype=object))).dropna().unique().tolist())
options = [{'label':'All Sites','value':'ALL'}] + [{'label':s,'value':s} for s in sites]

app.layout = html.Div([
    html.H2("SpaceX Debug Dashboard", style={'textAlign':'center'}),
    html.Div([
        html.Div([html.B("Data diagnostics (from CSV):")]),
        html.Pre(id='diag', style={'whiteSpace':'pre-wrap','fontSize':12})
    ], style={'width':'80%','margin':'auto','padding':'10px','border':'1px solid #ddd','background':'#f9f9f9'}),

    html.Br(),

    html.Div([
        dcc.Dropdown(id='site-dropdown', options=options, value='ALL', placeholder="Select a Launch Site here", searchable=True, clearable=False, style={'width':'60%'}),
    ], style={'display':'flex','justifyContent':'center'}),

    html.Br(),
    dcc.Graph(id='success-pie-chart'),
    html.Br(),

    html.Div([
        html.Label("Payload range (Kg):"),
        dcc.RangeSlider(id='payload-slider', min=0, max=10000, step=100, value=[min_payload,max_payload],
                        marks={0:'0',2500:'2500',5000:'5000',7500:'7500',10000:'10000'})
    ], style={'width':'80%','margin':'auto'}),

    html.Br(),
    dcc.Graph(id='success-payload-scatter-chart'),

    html.Br(),
    html.Div("Terminal logs will also show dataset diagnostics.", style={'textAlign':'center','color':'gray'})
])

# diag callback to display dataset info in the webpage
@app.callback(Output('diag','children'), Input('site-dropdown','value'))
def show_diag(_):
    s = []
    s.append(f"Rows: {df.shape[0]}")
    s.append(f"Columns: {df.columns.tolist()}")
    # show first 5 rows
    if not df.empty:
        s.append("\nFirst 5 rows:")
        s.append(df.head(5).to_string())
    # show class counts
    if 'class' in df.columns:
        s.append("\nclass value counts:")
        s.append(str(df['class'].value_counts(dropna=False).to_dict()))
    else:
        s.append("\nNo 'class' column found.")
    s.append("\nLaunch Site unique (first 20):")
    s.append(str(pd.Series(df.get('Launch Site', pd.Series([]))).dropna().unique()[:20].tolist()))
    s.append(f"\nPayload min/max (coerced numeric): {pd.to_numeric(df.get('Payload Mass (kg)', pd.Series([])), errors='coerce').min()} / {pd.to_numeric(df.get('Payload Mass (kg)', pd.Series([])), errors='coerce').max()}")
    return "\n".join(map(str,s))

# pie callback (safe)
@app.callback(Output('success-pie-chart','figure'), Input('site-dropdown','value'))
def pie(selected_site):
    if df.empty or 'class' not in df.columns:
        return px.pie(names=['No data'], values=[1], title='No data for pie')
    if selected_site=='ALL':
        grouped = df[df['class']==1].groupby('Launch Site').size().reset_index(name='success_count')
        if grouped.empty:
            grouped = df.groupby('Launch Site').size().reset_index(name='count')
            return px.pie(grouped, names='Launch Site', values='count', title='Launch counts per site')
        else:
            return px.pie(grouped, names='Launch Site', values='success_count', title='Successful launches by site')
    else:
        d = df[df['Launch Site']==selected_site]
        if d.empty:
            return px.pie(names=['No data'], values=[1], title=f'No data for {selected_site}')
        counts = d['class'].value_counts().reset_index()
        counts.columns=['class','count']
        counts['label']=counts['class'].map({1:'Success',0:'Failure'}).fillna(counts['class'].astype(str))
        return px.pie(counts, names='label', values='count', title=f'Success vs Failure {selected_site}')

# scatter callback (safe)
@app.callback(Output('success-payload-scatter-chart','figure'), [Input('site-dropdown','value'), Input('payload-slider','value')])
def scatter(selected_site, payload_range):
    low, high = payload_range if payload_range and len(payload_range)==2 else (min_payload,max_payload)
    dff = df.copy()
    # coerce payload to numeric
    dff['Payload Num'] = pd.to_numeric(dff.get('Payload Mass (kg)', pd.Series([])), errors='coerce')
    dff = dff[(dff['Payload Num']>=low) & (dff['Payload Num']<=high)]
    if selected_site!='ALL':
        dff = dff[dff['Launch Site']==selected_site]
    if dff.empty or 'class' not in dff.columns:
        return px.scatter(x=[0], y=[0], title='No data to display')
    color_col = 'Booster Version Category' if 'Booster Version Category' in dff.columns else None
    if color_col is None:
        dff['Booster Version Category'] = 'Unknown'
        color_col = 'Booster Version Category'
    fig = px.scatter(dff, x='Payload Num', y='class', color=color_col, hover_data=['Launch Site'])
    fig.update_xaxes(title='Payload Mass (kg)')
    fig.update_yaxes(tickvals=[0,1], ticktext=['Failure','Success'])
    return fig

if __name__=='__main__':
    print("[INFO] Starting debug app. If blank charts appear, check the diag box on page and terminal logs.", file=sys.stderr)
    app.run(debug=True, port=8050)
