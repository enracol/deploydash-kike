# -*- coding: utf-8 -*-
"""
Created on Tue May 29 14:56:58 2018

@author: n34873
"""

import os

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas_datareader.data as web
import plotly.graph_objs as go
import pandas as pd
import datetime
import math
from scipy.integrate import quad

#Version deploy################################################################
def dN(x):
    ''' Probability density function of standard normal random variable x. '''
    return math.exp(-0.5 * x ** 2) / math.sqrt(2 * math.pi)


def N(d):
    ''' Cumulative density function of standard normal random variable x. '''
    return quad(lambda x: dN(x), -20, d, limit=50)[0]


def d1f(St, K, t, T, r, sigma):
    ''' Black-Scholes-Merton d1 function.
        Parameters see e.g. BSM_call_value function. '''
    d1 = (math.log(St / K) + (r + 0.5 * sigma ** 2)
          * (T - t)) / (sigma * math.sqrt(T - t))
    return d1

#
# Valuation Functions
#


def BSM_call_value(St, K, t, T, r, sigma):
    d1 = d1f(St, K, t, T, r, sigma)
    d2 = d1 - sigma * math.sqrt(T - t)
    call_value = St * N(d1) - math.exp(-r * (T - t)) * K * N(d2)
    return call_value
###############################################################################





strikes = np.linspace(4000, 12000, 10) #no sé por qué no funciona el marks del slider cuando pongo 10

app = dash.Dash(__name__)
server = app.server

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div(children=[

    html.Div([
        html.Div([
            html.Label('Volatilidad',style={'width':'10%'}),
            dcc.RadioItems(
                    id='dropdown-a',
                    options=[{'label': i, 'value': i} for i in [0.20, 0.30, 0.40]],
                    value=0.20),
            ]),
        
        html.Label('Strike'),
        html.Div(dcc.Slider(
            id='input-escalar',
            min=4000,
            max=12000,
            marks={
                #3: {k: 'K= {}'.format(k) for k in strikes},
                4000:'K=4000',
                12000:'K=12000'
            },
            #step=1000,
            value=8000,
            #included = True,
            #vertical= True,
        ),style={'width': '40%'}),
        
        html.Div(html.Label('_')),
                
        html.Div([
            html.Label('Fecha a maturity'),
            dcc.Input(id='input-date', value='1')
            ]),
        
        html.Div([
            html.Label(children='Cotizaciones'),
            html.Div(dcc.RadioItems(
                    id='dropdown-b',
                    options=[{'label': i, 'value': i} for i in ['AMS', 'TSLA', 'SAN','AAPL','GASN']],
                    value='SAN')),
            ]),
        
        html.Div(id='example-graph'),
        
    ], style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),   
    
    
    html.Div([
        dcc.Graph(id='example-graph-b'),
        html.Div(id='example-graph-c'),
        ],style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),
])

@app.callback(Output('example-graph','children'),[Input( 'dropdown-a', 'value'),Input('input-escalar','value'),Input('input-date','value')])

def update_graph(selected_vol,escalar,fecha):
    
    K = float(escalar) # strike price
    T = float(fecha)  # time-to-maturity
    r = 0.025  # constant, risk-less short rate
    vol = selected_vol  # constant volatility
    S = np.linspace(4000, 12000, 150)  # vector of index level values
    C = [BSM_call_value(S0, K, 0, T, r, vol) for S0 in S]
    h = np.maximum(S - K, 0)
    
    return dcc.Graph(
        id='example-graphh',
        figure={
            'data': [
                {'x': S,'y': C,'type': 'line','name': 'Valor_BS'},
                {'x': S,'y': h,'type': 'line','name': 'Valor_Intrinseco'},
            ],
            'layout': {
                'title': 'Strike {}'.format(K)
            }
        }
    )
        
@app.callback(Output('example-graph-b','figure'),[Input( 'dropdown-b', 'value')])

def update_graphh(selected_stock):
        
    hoy=datetime.datetime.now()
    start= hoy.replace(hoy.year-6,hoy.month,hoy.day+1) #'iex' solo almacena 6 años de datos desde la fecha de hoy
        
    df=web.DataReader(selected_stock,'iex',start,hoy)
    df2=web.DataReader('SAN','iex',start,hoy)
    
    return {
        'data': [
            {'x': df.index,'y': df.close[:,],'type': 'line','name': selected_stock},
            {'x': df2.index,'y': df2.close[:,],'type': 'line','name': 'SAN'},
            ]
        }

@app.callback(Output('example-graph-c','children'),[Input('example-graph-b','hoverData'),Input( 'dropdown-a', 'value'),Input('input-escalar','value'),Input('input-date','value')])

def update_graphhhh(hover_S,selected_vol,escalar,fecha):
    
    K = float(escalar) # strike price
    T = float(fecha)  # time-to-maturity
    r = 0.025  # constant, risk-less short rate
    vol = selected_vol  # constant volatility
    S_0 = hover_S['points'][0]['y']
    S_1 = hover_S['points'][1]['y']
    #trace_name_0=app.layout['example-graph-b'].figure['data'][0]['name'] #Creo que no vale porque example-graph-b no tiene definido un figure
    #trace_name_1=app.layout['example-graph-b'].figure['data'][1]['name']
    
    title= 'Precio Call en la fecha: {}'.format(hover_S['points'][0]['x'])
    
    C_0 = BSM_call_value(S_0, S_0*K/8000, 0, T, r, vol)
    C_1 = BSM_call_value(S_1, S_1*K/8000, 0, T, r, vol)            
    
    return dcc.Graph(
        id='example-graphhhh',
        figure={
            'data':[
                {'x': [S_0],'y': [C_0],'type': 'bar','name': 'trace_name_0'},
                {'x': [S_1],'y': [C_1],'type': 'bar','name': 'trace_name_1'},
            ],
            'layout': {'title': title}
            }
    )

if __name__ == '__main__':
    app.run_server()