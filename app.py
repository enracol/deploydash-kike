# -*- coding: utf-8 -*-
"""
@author: Enrique Ramos
"""

import os

import numpy as np
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd

pd.core.common.is_list_like = pd.api.types.is_list_like

import pandas_datareader.data as web
import plotly.graph_objs as go
import datetime
import math
from scipy.integrate import quad

app = dash.Dash(__name__)
server = app.server

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

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
mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
app.scripts.append_script({ 'external_url' : mathjax })

markdown_text = '''
### Black-Scholes
El modelo de Black-Scholes o ecuación de Black-Scholes es una ecuación usada en matemática financiera para determinar el precio de determinados activos financieros. Dicha ecuación se basa ampliamente en la teoría de procesos estocásticos en particular modela variaciones de precios como un proceso de Wiener. En 1973, Robert C. Merton publicó "Theory of Rational Option Pricing", en él hacía referencia a un modelo matemático desarrollado por Fisher Black y Myron Scholes.
A este modelo lo denominó Black-Scholes y fue empleado para estimar el valor actual de una opción europea para la compra (Call), o venta (Put), de acciones en una fecha futura. Posteriormente el modelo se amplió para opciones sobre acciones que producen dividendos, y luego se adoptó para opciones europeas, americanas, y mercado monetario. Los modelos de valoración de opciones son también aplicados actualmente a la valoración de activos intangibles, tales como patentes.
 * Fuente: [Enlace Wikipedia](https://es.wikipedia.org/wiki/Modelo_de_Black-Scholes)
 * En fórmulas:    
     $$C = S\Phi(d_{+}) - \\text{K} e^{-r_{d}T}\Phi(d_{-})$$
     $$P = -S\Phi(-d_{+}) + K e^{-r_{d}T}\Phi(-d_{-})$$
donde
     $$d_{\\pm} =\dfrac{ ln(S/K)+(r_{d}-r_{e} \\pm \sigma^{2}/2)T}{\sigma\sqrt{T}}$$
'''
markdown_text_2 = '''
### Call con precios históricos.
Selecciona una acción. Pudes ver el precio histórico de cotización para la acción seleccionada comparada contra la de Banco Santander.
Poniéndote con el ratón encima de la gráfica, se actualizará el valor de la opción en la gráfica de la derecha para esa fecha
'''
strikes = np.linspace(4000, 12000, 10) #no sé por qué no funciona el marks del slider cuando pongo 10


app.layout = html.Div(children=[

    html.Div([
        dcc.Markdown(children=markdown_text),
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
        ),style={'width': '49%'}),
        
        html.Div([
            html.Label('Fecha a maturity'),
            dcc.Input(id='input-date', value='1')
            ],style={'display': 'inline-block','float': 'right', 'width': '49%'}),
            
    ],style={'display': 'inline-block', 'width': '49%'}),
    
    html.Div(id='example-graph',style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),
    
    dcc.Markdown(children=markdown_text_2),
    html.Div([
            html.Label(children='Cotizaciones'),
            html.Div(dcc.RadioItems(
                    id='dropdown-b',
                    options=[{'label': i, 'value': i} for i in ['AMS', 'TSLA', 'SAN','AAPL']],
                    value='SAN'))
            ]),
    
    dcc.Graph(id='example-graph-b',style={'display': 'inline-block','width': '49%'}),
    html.Div(id='example-graph-c',style={'display': 'inline-block', 'float': 'right', 'width': '49%'}),
    
    #html.Div([
    #    dcc.Graph(id='example-graph-b'),
    #    html.Div(id='example-graph-c'),
    #    ],style={'display': 'inline-block', 'width': '49%'}),

    
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
    start= hoy.replace(hoy.year-5,hoy.month,hoy.day) #'iex' solo almacena 6 años de datos desde la fecha de hoy
        
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
            'layout': {'title': title, 'yaxis': dict(range=[0, 3])}
        }
    )
        

if __name__ == '__main__':
    app.run_server()
