import io
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy.optimize import minimize
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, accuracy_score, confusion_matrix

st.set_page_config(page_title='BVL | SVR + Markowitz', layout='wide')
st.title('BVL: predicción SVR y frontera eficiente de Markowitz')
st.caption('Prototipo académico. Predicciones inciertas; no constituye recomendación de inversión.')

ASSETS = {'Buenaventura':'BUENAVC1','Cerro Verde':'CVERDEC1','UNACEM':'UNACEMC1',
          'El Brocal':'BROCALC1','Siderperú':'SIDERC1','Engie Perú':'ENGIEC1',
          'InRetail':'INRETC1','Casa Grande':'CASAGRC1','BBVA Perú':'BBVAC1',
          'Aceros Arequipa':'CORAREC1'}
TICKERS = list(ASSETS.values())

@st.cache_data(show_spinner=False)
def demo_data():
    rng=np.random.default_rng(42)
    dates=pd.bdate_range('2021-01-04',periods=1100)
    market=rng.normal(0.00025,0.010,len(dates))
    frames=[]
    for i,t in enumerate(TICKERS):
        rets=0.0001+(0.65+0.1*i)*market+rng.normal(0,0.007+0.001*i,len(dates))
        close=(5+i*2)*np.exp(np.cumsum(rets))
        op=close*np.exp(rng.normal(0,0.003,len(dates)))
        high=np.maximum(op,close)*(1+rng.uniform(0.001,0.012,len(dates)))
        low=np.minimum(op,close)*(1-rng.uniform(0.001,0.012,len(dates)))
        vol=rng.integers(1000,100000,len(dates))
        frames.append(pd.DataFrame({'Date':dates,'Ticker':t,'Open':op,'High':high,'Low':low,'Close':close,'Volume':vol}))
    return pd.concat(frames,ignore_index=True)

def clean_data(raw):
    aliases={'date':'Date','fecha':'Date','ticker':'Ticker','simbolo':'Ticker',
             'open':'Open','apertura':'Open','high':'High','maximo':'High',
             'low':'Low','minimo':'Low','close':'Close','cierre':'Close','volume':'Volume','volumen':'Volume'}
    d=raw.copy(); d.columns=[aliases.get(str(c).strip().lower(),str(c).strip()) for c in d.columns]
    required=['Date','Ticker','Open','High','Low','Close','Volume']
    missing=set(required)-set(d.columns)
    if missing: raise ValueError('Columnas faltantes: '+', '.join(sorted(missing)))
    d=d[required].copy();d['Date']=pd.to_datetime(d['Date'],errors='coerce')
    d['Ticker']=d['Ticker'].astype(str).str.strip().str.upper()
    for c in ['Open','High','Low','Close','Volume']:
        d[c]=pd.to_numeric(d[c],errors='coerce')
    d=d.dropna(subset=required).sort_values(['Ticker','Date']).drop_duplicates(['Ticker','Date'],keep='last')
    d=d[(d[['Open','High','Low','Close']]>0).all(axis=1)&(d['Volume']>=0)]
    return d

def features(d):
    x=d.sort_values('Date').copy(); c=x['Close'];h=x['High'];l=x['Low'];v=x['Volume']
    delta=c.diff(); up=delta.clip(lower=0).ewm(alpha=1/14,adjust=False).mean();down=(-delta.clip(upper=0)).ewm(alpha=1/14,adjust=False).mean()
    rs=up/down.replace(0,np.nan);x['RSI']=100-100/(1+rs)
    x['MACD']=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    x['ROC']=c.pct_change(12)
    weights=np.arange(1,15,dtype=float)
    x['WMA']=c.rolling(14).apply(lambda a: np.dot(a,weights)/weights.sum(),raw=True)/c-1
    x['OBV']=((np.sign(c.diff()).fillna(0))*v).cumsum()
    x['OBV_change']=x['OBV'].diff(5)/(v.rolling(20).mean()*5).replace(0,np.nan)
    lo=l.rolling(14).min();hi=h.rolling(14).max();span=(hi-lo).replace(0,np.nan)
    x['WilliamsR']=-100*(hi-c)/span
    x['Stochastic']=100*(c-lo)/span
    x['Return1']=c.pct_change()
    x['TargetReturn']=c.shift(-1)/c-1
    x['TargetPrice']=c.shift(-1)
    return x.replace([np.inf,-np.inf],np.nan)

FEATURES=['RSI','MACD','ROC','WMA','OBV_change','WilliamsR','Stochastic','Return1']

def forecast_one(d,train_ratio=0.8):
    f=features(d)
    labeled=f.dropna(subset=FEATURES+['TargetReturn','TargetPrice']).copy()
    if len(labeled)<130: raise ValueError('Se requieren al menos 130 observaciones con indicadores completos')
    cut=int(len(labeled)*train_ratio)
    tr=labeled.iloc[:cut];te=labeled.iloc[cut:]
    model=make_pipeline(StandardScaler(),SVR(kernel='linear',C=0.1,epsilon=0.005))
    model.fit(tr[FEATURES],tr['TargetReturn'])
    pred=model.predict(te[FEATURES]);truth=te['TargetReturn'].to_numpy()
    last=f.dropna(subset=FEATURES).iloc[-1]
    next_ret=float(model.predict(last[FEATURES].to_frame().T)[0])
    yhat=te['Close'].to_numpy()*(1+pred)
    metrics={'RMSE precio':float(np.sqrt(mean_squared_error(te['TargetPrice'],yhat))),
             'MAPE precio (%)':float(mean_absolute_percentage_error(te['TargetPrice'],yhat)*100),
             'Accuracy dirección':float(accuracy_score(truth>0,pred>0)),
             'N test':len(te)}
    return next_ret,metrics,te,pred,truth

def optimize(mu,cov,cap=1.0,mode='sharpe',rf=0.03,target=None):
    n=len(mu); bounds=[(0,cap)]*n; cons=[{'type':'eq','fun':lambda w:np.sum(w)-1}]
    if target is not None: cons.append({'type':'eq','fun':lambda w:float(w@mu-target)})
    def risk(w): return float(np.sqrt(max(w@cov@w,0)))
    def objective(w):
        if mode=='min':return float(w@cov@w)
        return -float((w@mu-rf)/max(risk(w),1e-9))
    # SLSQP, with several deterministic starting points
    starts=[np.ones(n)/n]+[np.eye(n)[i]*0.8+np.ones(n)*0.2/n for i in range(n)]
    best=None
    for x0 in starts:
        result=minimize(objective,x0,method='SLSQP',bounds=bounds,constraints=cons,
                        options={'maxiter':1500,'ftol':1e-11})
        if result.success and (best is None or result.fun<best.fun):best=result
    return None if best is None else best.x

with st.sidebar:
    st.header('Configuración')
    source=st.radio('Fuente de datos',['CSV propio (datos reales)','Datos sintéticos (demostración)'])
    uploaded=st.file_uploader('CSV OHLCV (formato largo)',type=['csv']) if source.startswith('CSV') else None
    st.caption('Columnas: Date, Ticker, Open, High, Low, Close, Volume. Un registro por fecha y acción.')
    chosen=st.multiselect('Activos a analizar',TICKERS,default=TICKERS)
    train_pct=st.slider('Porcentaje entrenamiento SVR',60,90,80,5)
    max_weight=st.slider('Máximo por activo (%)',10,100,35,5)/100
    rf=st.number_input('Tasa libre de riesgo anual (%)',value=3.0,step=0.5)/100
    blend=st.slider('Peso del pronóstico SVR en retorno esperado (%)',0,100,30,5)/100
    horizon=st.selectbox('Horizonte de pronóstico',['Siguiente sesión bursátil'],index=0)
    st.caption('SVR predice retorno de la siguiente sesión. Markowitz usa retornos anualizados.')

if source.startswith('CSV') and uploaded is None:
    st.info('Suba un CSV OHLCV para trabajar con cotizaciones reales. También puede seleccionar datos sintéticos para explorar la interfaz.')
    st.download_button('Descargar CSV de ejemplo sintético',demo_data().to_csv(index=False).encode(),file_name='ejemplo_sintetico_bvl.csv',mime='text/csv')
    st.stop()
try:
    raw=pd.read_csv(uploaded) if uploaded is not None else demo_data()
    data=clean_data(raw)
except Exception as e:
    st.error(f'No se pudieron procesar los datos: {e}');st.stop()
if source.startswith('Datos'):
    st.warning('MODO DEMOSTRACIÓN: los precios son completamente sintéticos; NO son cotizaciones de la BVL.')
if len(chosen)<2:
    st.error('Seleccione al menos dos acciones.');st.stop()
if max_weight*len(chosen)<1-1e-8:
    st.error('Restricción imposible: número de acciones × peso máximo debe ser al menos 100%.');st.stop()

data=data[data['Ticker'].isin(chosen)]
rows=[];predict={}; details={}
for ticker in chosen:
    d=data[data['Ticker']==ticker]
    try:
        pred,m,te,predtest,truth=forecast_one(d,train_pct/100)
        rows.append({'Ticker':ticker,'Último cierre':float(d.iloc[-1]['Close']),
                     'Predicción retorno próximo día':pred,'Dirección':'SUBE' if pred>0 else 'BAJA',**m})
        predict[ticker]=pred;details[ticker]=(te,predtest,truth)
    except Exception as e:st.warning(f'{ticker}: excluido ({e})')
if len(rows)<2:st.error('Menos de dos activos con datos válidos.');st.stop()
summary=pd.DataFrame(rows).set_index('Ticker')
# Use actual coincident trading dates; never forward-fill prices or interpolate illiquid trading gaps.
prices=data[data['Ticker'].isin(summary.index)].pivot(index='Date',columns='Ticker',values='Close').sort_index()
prices=prices.dropna(how='any')
if len(prices)<130:st.error('Hay menos de 130 fechas con precios observados para todos los activos.');st.stop()
returns=prices.pct_change(fill_method=None).dropna()
if len(returns)<100:st.error('No hay suficientes retornos comunes.');st.stop()
mu_hist=returns.mean()*252
mu_svr=pd.Series(predict).reindex(returns.columns)*252
# Annualized one-day SVR projection is an assumption, not a multi-day forecast.
mu=((1-blend)*mu_hist+blend*mu_svr).to_numpy()
cov=(returns.cov()*252).to_numpy();cov=(cov+cov.T)/2+np.eye(len(mu))*1e-8
cap=max_weight
wmin=optimize(mu,cov,cap,mode='min')
wsh=optimize(mu,cov,cap,mode='sharpe',rf=rf)
if wmin is None or wsh is None:st.error('Optimización no convergió. Revise restricciones y datos.');st.stop()

def stats(w):
    ret=float(w@mu);vol=float(np.sqrt(w@cov@w));return ret,vol,(ret-rf)/vol if vol>0 else np.nan

st.subheader('1. Predicción de tendencia (SVR)')
st.dataframe(summary.style.format({'Último cierre':'{:.3f}','Predicción retorno próximo día':'{:.2%}',
 'RMSE precio':'{:.3f}','MAPE precio (%)':'{:.2f}','Accuracy dirección':'{:.1%}'}),use_container_width=True)
st.caption('Evaluación cronológica fuera de muestra. La precisión direccional se calcula sobre el conjunto de prueba; no garantiza desempeño futuro.')
asset=st.selectbox('Inspeccionar validación de un activo',summary.index)
te,predtest,truth=details[asset]
figp=go.Figure()
figp.add_trace(go.Scatter(x=te['Date'],y=te['TargetPrice'],name='Precio real siguiente sesión'))
figp.add_trace(go.Scatter(x=te['Date'],y=te['Close']*(1+predtest),name='Predicción SVR'))
figp.update_layout(title=f'Validación temporal de {asset}',xaxis_title='Fecha de señal',yaxis_title='Precio',height=340)
st.plotly_chart(figp,use_container_width=True)
cm=confusion_matrix(truth>0,predtest>0,labels=[False,True])
st.write('Matriz de confusión direccional (filas: real BAJA/SUBE; columnas: predicho BAJA/SUBE)')
st.dataframe(pd.DataFrame(cm,index=['Real BAJA','Real SUBE'],columns=['Pred BAJA','Pred SUBE']))

st.subheader('2. Frontera eficiente de Markowitz')
# Generate attainable efficient points using a return target between minimum-variance and max-return portfolios.
retmin,volmin,_=stats(wmin)
# Max return linear objective under long-only caps: allocate to highest expected returns.
order=np.argsort(-mu);wmax=np.zeros(len(mu));remain=1.
for idx in order:
    amount=min(cap,remain);wmax[idx]=amount;remain-=amount
    if remain<=1e-10:break
retmax=float(wmax@mu)
points=[]
for target in np.linspace(retmin,retmax,55):
    w=optimize(mu,cov,cap,mode='min',target=float(target))
    if w is not None:
        r,v,s=stats(w);points.append((v,r,s))
fig=go.Figure()
if points:
    p=np.array(points)
    fig.add_trace(go.Scatter(x=p[:,0]*100,y=p[:,1]*100,mode='lines+markers',
                             marker=dict(size=5),name='Frontera eficiente'))
for label,w,symbol in [('Mínima varianza',wmin,'diamond'),('Máximo Sharpe',wsh,'star')]:
    r,v,s=stats(w)
    fig.add_trace(go.Scatter(x=[v*100],y=[r*100],mode='markers+text',text=[label],
                             textposition='top center',marker=dict(size=16,symbol=symbol),name=label))
fig.update_layout(xaxis_title='Volatilidad anualizada (%)',yaxis_title='Retorno esperado anualizado (%)',
                  height=540,legend=dict(orientation='h',y=-0.2))
st.plotly_chart(fig,use_container_width=True)

choice=st.radio('Portafolio a presentar',['Máximo Sharpe','Mínima varianza'],horizontal=True)
w=wsh if choice=='Máximo Sharpe' else wmin
r,v,s=stats(w)
a,b,c=st.columns(3)
a.metric('Retorno esperado anualizado',f'{r:.2%}')
b.metric('Volatilidad anualizada',f'{v:.2%}')
c.metric('Ratio Sharpe',f'{s:.2f}')
weights=pd.DataFrame({'Ticker':returns.columns,'Peso':w}).sort_values('Peso',ascending=False)
st.subheader('3. Composición óptima')
st.bar_chart(weights.set_index('Ticker')['Peso'])
st.dataframe(weights.style.format({'Peso':'{:.2%}'}),use_container_width=True,hide_index=True)
st.download_button('Descargar asignación óptima CSV',weights.to_csv(index=False).encode('utf-8'),
                   file_name='portafolio_markowitz.csv',mime='text/csv')
st.subheader('4. Supuestos y limitaciones')
st.markdown('''- **Datos:** el modo CSV exige OHLCV real y símbolos identificados. Los datos sintéticos son exclusivamente didácticos.
- **Horizonte:** SVR estima el retorno de **una sesión**. Multiplicarlo por 252 para construir un retorno anual es una extrapolación fuerte y potencialmente inestable; ajuste el peso SVR a 0% para usar solo medias históricas.
- **Markowitz:** cartera long-only, completamente invertida, sin apalancamiento, comisiones, impuestos ni límites de liquidez; matriz de covarianzas histórica anualizada.
- **Calendario:** solo se utilizan fechas comunes con precios observados; no se inventan operaciones en días sin negociación.
- **Validación:** separación cronológica simple, no backtesting walk-forward ni evaluación de costos de transacción. La BVL puede presentar activos poco líquidos.
- **Moneda:** mezclar acciones cotizadas en PEN y USD exige convertirlas a una moneda común antes de cargar los datos. El CSV debe estar normalizado a una moneda base.
- **Dividendos y eventos corporativos:** idealmente utilizar precios ajustados o retornos totales consistentes.
''')
