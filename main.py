import os, requests, numpy as np, pandas as pd, ta, pytz
from datetime import datetime, time, timedelta
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import ApplicationBuilder, CommandHandler, CallbackQueryHandler, ContextTypes, JobQueue

TOKEN = "8354975270:AAE-IiNUXCT42UpqH_0Z4tusKMQ8N64EwPE"
NEWS_API = "332bf45035354091b59f1f64601e2e11"
FX_API = "ca1acbf0cedb4488b130c59252891c"

MODEL_PATH = "ai_model_portfolio.h5"
TRAIN_LOG = "last_train_portfolio.txt"

CRYPTO = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT"]
FOREX = ["EURUSD","GBPUSD","USDJPY","XAUUSD"]
UTC = pytz.UTC

SESSIONS = {
    "London": (time(8,0), time(16,0)),
    "NewYork": (time(13,0), time(21,0))
}

portfolio = {}

def current_session():
    now = datetime.now(UTC).time()
    for k,(a,b) in SESSIONS.items():
        if a <= now <= b:
            return k
    return None

def news_blackout(symbol):
    try:
        q = symbol.replace("USDT","")
        r = requests.get("https://newsapi.org/v2/everything",
                         params={"q":q,"language":"en","apiKey":NEWS_API,"pageSize":10},
                         timeout=10).json()
        blockers=["cpi","rate","inflation","powell","fed","ecb","nfp","interest"]
        return any(any(k in a["title"].lower() for k in blockers) for a in r.get("articles",[]))
    except:
        return True

def crypto_data(sym,tf,l=300):
    try:
        r = requests.get("https://api.binance.com/api/v3/klines",
                     params={"symbol":sym,"interval":tf,"limit":l},timeout=10).json()
        df = pd.DataFrame(r, columns=list("tohlcv")+["x"]*6)
        df[["o","h","l","c","v"]] = df[["o","h","l","c","v"]].astype(float)
        return df
    except:
        return pd.DataFrame(columns=["o","h","l","c","v"])

def forex_data(pair,tf):
    try:
        r = requests.get("https://www.alphavantage.co/query",
                         params={"function":"FX_INTRADAY",
                                 "from_symbol":pair[:3],
                                 "to_symbol":pair[3:],
                                 "interval":tf,
                                 "apikey":FX_API},timeout=10).json()
        ts=[v for k,v in r.items() if "Time Series" in k]
        if not ts: return pd.DataFrame(columns=["o","h","l","c"])
        df=pd.DataFrame(ts[0]).T.astype(float)
        df.rename(columns={"1. open":"o","2. high":"h","3. low":"l","4. close":"c"}, inplace=True)
        return df
    except:
        return pd.DataFrame(columns=["o","h","l","c"])

def enrich(df):
    if len(df)==0: return df
    try:
        df["EMA20"]=ta.trend.EMAIndicator(df["c"],20).ema_indicator()
        df["EMA50"]=ta.trend.EMAIndicator(df["c"],50).ema_indicator()
        df["RSI"]=ta.momentum.RSIIndicator(df["c"],14).rsi()
        macd=ta.trend.MACD(df["c"])
        df["MACD"], df["MS"]=macd.macd(), macd.macd_signal()
        df["ATR"]=ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range()
        df["VWAP"]=(df["v"]*(df["h"]+df["l"]+df["c"])/3).cumsum()/df["v"].cumsum()
    except:
        pass
    return df

def liquidity_sweep(df):
    if len(df)<10: return None
    hi,lo=df["h"].iloc[-1],df["l"].iloc[-1]
    prev_hi,prev_lo=df["h"].iloc[-10:-1].max(),df["l"].iloc[-10:-1].min()
    if hi>prev_hi: return "buy"
    if lo<prev_lo: return "sell"
    return None

def order_block(df,direction):
    if len(df)==0: return 0
    if direction=="BUY": return df["l"].tail(20).min()
    return df["h"].tail(20).max()

def hedge_logic(htf,ltf,style):
    if len(htf)==0 or len(ltf)==0: return None
    h,l = htf.iloc[-1], ltf.iloc[-1]
    bias=liquidity_sweep(ltf)
    trend_up=h.get("EMA20",0)>h.get("EMA50",0)
    trend_dn=h.get("EMA20",0)<h.get("EMA50",0)
    vol=l.get("ATR",0)>ltf["ATR"].rolling(50).mean().iloc[-1] if len(ltf)>=50 else True
    if trend_up and l.get("c",0)>l.get("VWAP",0) and l.get("RSI",0)>55 and l.get("MACD",0)>l.get("MS",0) and bias!="sell":
        sl=order_block(ltf,"BUY"); e=l["c"]; tp=e+(e-sl)*(2 if style=="swing" else 1.5)
        return "BUY",e,sl,tp,vol
    if trend_dn and l.get("c",0)<l.get("VWAP",0) and l.get("RSI",0)<45 and l.get("MACD",0)<l.get("MS",0) and bias!="buy":
        sl=order_block(ltf,"SELL"); e=l["c"]; tp=e-(sl-e)*(2 if style=="swing" else 1.5)
        return "SELL",e,sl,tp,vol
    return None

def news_sentiment(symbol):
    try:
        q=symbol.replace("USDT","")
        r=requests.get("https://newsapi.org/v2/everything",
                       params={"q":q,"language":"en","apiKey":NEWS_API,"pageSize":5},
                       timeout=10).json()
        score=0
        for a in r.get("articles",[]):
            score+=TextBlob(a.get("title","")).sentiment.polarity
        return score/max(len(r.get("articles",[])),1)
    except: return 0

class MarketAI:
    def __init__(self, window=30):  
        self.window=window
        self.scaler=MinMaxScaler()
        self.model=self.load_or_create()

    def load_or_create(self):
        if os.path.exists(MODEL_PATH): 
            try: return load_model(MODEL_PATH)
            except: os.remove(MODEL_PATH)
        model=Sequential([
            LSTM(64,return_sequences=True,input_shape=(self.window,5)),  
            Dropout(0.2),
            LSTM(32),
            Dense(1,activation="sigmoid")
        ])
        model.compile(optimizer="adam", loss="binary_crossentropy")
        return model

    def features(self,df):
        df=df.copy(); df["r"]=df["c"].pct_change()
        df["v"]=df["r"].rolling(10).std()
        df["rsi"]=ta.momentum.RSIIndicator(df["c"],14).rsi()
        df["ema"]=ta.trend.EMAIndicator(df["c"],20).ema_indicator()
        df["ed"]=df["c"]-df["ema"]
        df=df.dropna()
        return df[["r","v","rsi","ed","c"]]

    def prepare(self,df):
        f=self.features(df)
        s=self.scaler.fit_transform(f)
        X,y=[],[]
        for i in range(self.window,len(s)-1):
            X.append(s[i-self.window:i])
            y.append(1 if f["c"].iloc[i+1]>f["c"].iloc[i] else 0)
        return np.array(X,dtype=np.float32), np.array(y,dtype=np.float32)

    def train_daily(self,df):
        if os.path.exists(TRAIN_LOG):
            try:
                last=datetime.fromisoformat(open(TRAIN_LOG).read())
                if datetime.utcnow()-last<timedelta(hours=24): return
            except: pass
        X,y=self.prepare(df)
        if len(X)==0: return
        self.model.fit(X,y,epochs=4,batch_size=8,verbose=0)
        self.model.save(MODEL_PATH)
        open(TRAIN_LOG,"w").write(str(datetime.utcnow()))

    def predict(self,df):
        f=self.features(df)
        s=self.scaler.transform(f)
        if len(s)<self.window: return None
        X=np.array([s[-self.window:]],dtype=np.float32)
        return float(self.model.predict(X,verbose=0)[0][0])

class RLTrader:
    def decide(self, prob, news):
        score = abs(prob-0.5)*2 + abs(news)
        if score<0.75: return "NO TRADE", score
        return ("BUY" if prob>0.5 else "SELL"), score

async def start(update:Update,context:ContextTypes.DEFAULT_TYPE):
    kb=[[InlineKeyboardButton(a,callback_data=a)] for a in CRYPTO+FOREX]
    await update.message.reply_text("Select Asset for AI Trading",reply_markup=InlineKeyboardMarkup(kb))

async def analyze(update:Update,context:ContextTypes.DEFAULT_TYPE):
    q=update.callback_query
    if q: await q.answer()
    asset=q.data if q else None
    session=current_session()
    if session is None or asset is None:
        if q: await q.edit_message_text("SESSION CLOSED")
        return

    df=crypto_data(asset,"5m") if asset in CRYPTO else forex_data(asset,"5min")
    df=enrich(df)

    ai=MarketAI(); ai.train_daily(df)
    prob=ai.predict(df)
    if prob is None:
        if q: await q.edit_message_text("NOT ENOUGH DATA")
        return

    news=news_sentiment(asset)
    rl=RLTrader()
    decision,confidence_score=rl.decide(prob,news)
    if decision=="NO TRADE":
        if q: await q.edit_message_text("NO EDGE")
        return

    price=df["c"].iloc[-1]
    atr=ta.volatility.AverageTrueRange(df["h"],df["l"],df["c"],14).average_true_range().iloc[-1]
    sl=price-atr if decision=="BUY" else price+atr
    tp=price+atr*2 if decision=="BUY" else price-atr*2

    portfolio[asset]={"direction":decision,"entry":price,"SL":sl,"TP":tp,"confidence":confidence_score}

    explanation=f"🧠 AI Hedge Fund Trade Plan:\nAsset: {asset}\nDirection: {decision}\nEntry: {round(price,2)}\nSL: {round(sl,2)}\nTP: {round(tp,2)}\n"
    explanation+=f"AI Probability: {round(prob,3)}, News Sentiment: {round(news,3)}, Confidence: {round(confidence_score*100,1)}%\n"
    explanation+=f"EMA20: {round(df['EMA20'].iloc[-1],2)}, EMA50: {round(df['EMA50'].iloc[-1],2)}, RSI: {round(df['RSI'].iloc[-1],2)}, MACD: {round(df['MACD'].iloc[-1],2)}"
    if q: await q.edit_message_text(explanation)

if __name__=="__main__":
    jq = JobQueue(timezone=UTC)
    jq.start()
    app = ApplicationBuilder()\
        .token(TOKEN)\
        .job_queue(jq)\
        .build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CallbackQueryHandler(analyze))
    print("Bot running")
    app.run_polling()
