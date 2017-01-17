mport pandas as pd
import pymssql
import numpy as np
import string
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from scipy import stats
import time
import datetime
import pickle
import subprocess
from sklearn.externals.six import StringIO
#import pydotplus
import os
from IPython.display import Image
   
connSQL = pymssql.connect(server='pvwchi6psql1', user='sparky1', password='Sp@rk_users' , database='igtdev')

startTime = time.time()
def printfull(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')
   
def daysTo(arr):
    return min(np.nonzero(arr))
   
def rank(array):
    s = pd.Series(array)
    return s.rank(ascending=False)[len(s)-1]
   
def rank(array,value):
    s = pd.Series(array)
    return s.rank(ascending=False)[len(s)-1]


################################# Combine Dataframes ######################################
dfOptions = pd.read_pickle("L:\Matt Shared\pickles\OptionData.pkl")
####### Only Traded Options ######
##dfTradeData = pd.read_pickle("C:\Users\mmacfarlane\Desktop\MattSeven\Python\pickles\TradeData.pkl").reset_index()
##dfTradeData['StockSymbol'] = dfTradeData['StockSymbol'].str.strip()
##dfTradeData = dfTradeData[dfTradeData['Trader'] == 'Matt']
##tradedOptions = dfTradeData['OptionID'].drop_duplicates().tolist()
##dfOptions = dfOptions.reset_index()
##dfOptions = dfOptions[dfOptions['OptionID'].isin(tradedOptions)]
##dfOptions = dfOptions.set_index(['Date','OptionID'])
######
##dfOptions['Cost'] = dfOptions['Width'] / 2 * 100 / dfOptions['ClosePrice']
##dfOptions['IVA'] = dfOptions['ImpliedVolatility'] + (dfOptions['Width'] / 2 / dfOptions['Vega'])
##dfOptions['IVB'] = dfOptions['ImpliedVolatility'] - (dfOptions['Width'] / 2 / dfOptions['Vega'])
##dfOptions['IVWidth'] = dfOptions['IVA'] - dfOptions['IVB']
#dfOptions['1dIVchg'] = dfOptions['ImpliedVolatility'] - dfOptions['ImpliedVolatility'].groupby(level=['OptionID']).shift(-1)
#dfOptions['5dIVchg'] = dfOptions['ImpliedVolatility'] - dfOptions['ImpliedVolatility'].groupby(level=['OptionID']).shift(-5)
#dfOptions['10dIVchg'] = dfOptions['ImpliedVolatility'] - dfOptions['ImpliedVolatility'].groupby(level=['OptionID']).shift(-10)
##dfOptions['5dIVchgRank'] = dfOptions['5dIVchg'].groupby(level=['Date']).rank(pct=True)
##dfOptions['10dIVchgRank'] = dfOptions['10dIVchg'].groupby(level=['Date']).rank(pct=True)
##dfOptions['IVRank'] = dfOptions['ImpliedVolatility'].groupby(level=['Date']).rank(pct=True)
#dfOptions['d'] = np.where(dfOptions['Delta']<0,dfOptions['Delta']+1,dfOptions['Delta']) * 100
#dfOptions['AbsDelta'] = abs(dfOptions['Delta'])
#dfOptions['SkewAdj'] = .00000021572 * dfOptions['d'] ** 3 - .0000175063 * dfOptions['d'] ** 2 + .000935057 * dfOptions['d']
#
#dfOptions = dfOptions[dfOptions['AbsDelta']<.5]
#dfOptions = dfOptions[dfOptions['ImpliedVolatility']>.02]
#dfOptions = dfOptions[dfOptions['Width']>0]
#dfOptions = dfOptions.reset_index()
#dfOptions['StockSymbol'] = dfOptions['StockSymbol'].str.strip()
###dfOptions = dfOptions[['Date','OptionID','StockSymbol','ImpliedVolatility','Pnl','5dPnl','10dPnl','21dPnl','PnlToExp','d','SkewAdj','DTExp','CallPut','Vega','Mark','Delta']]
#
#dfStock = pd.read_pickle("C:\Users\mmacfarlane\Desktop\MattSeven\Python\pickles\StockData.pkl")
##dfStock['0%Vol'] = dfStock['HV5'].groupby(level=['StockSymbol']).apply(lambda x: x.rolling(window=252, min_periods=220).quantile(0))
#dfStock = dfStock.reset_index()
#ETFList = ['DIA','GDX','GDXJ','IWM','IYR','KBE','KRE','QQQ','SMH','SPY','XLB','XLE','XLF','XLK','XLP','XLV','XOP','XRT']
#dfStock['isETF'] = np.where(dfStock['StockSymbol'].isin(ETFList),1,0)
##dfStock = dfStock[['Date','StockSymbol','closePrice','Ern','DTErn','EMA20','HV5','XF','10dStock','isETF']]
#
##dfMarket = pd.read_excel('data\dfMarket.xlsx')
##dfMarket['Date'] = pd.to_datetime(dfMarket['Date'])
##dfSPY = dfStock[dfStock['StockSymbol']=='SPY'][['Date','10dStock']]
##dfSPY = dfSPY.rename(columns={"10dStock": "SPY10dStock"})
##dfMarket = dfMarket.merge(dfSPY, how='left', left_on=['Date'], right_on=['Date'])
##dfMarket = dfMarket[['Date','DTExpiration','DSExpiration','DayofWeek','SPY10dStock']]
#
####### Join Tables #############
##dfOptions = dfOptions[dfOptions['Date']>'2011-01-01']
#dfOptions = dfOptions.merge(dfStock, how='left', left_on=['Date','StockSymbol'], right_on=['Date','StockSymbol'])
##dfOptions = dfOptions.merge(dfMarket, how='left', left_on=['Date'], right_on=['Date'])
#dfOptions = dfOptions.set_index(['Date','StockSymbol','OptionID'])
##dfOptions.to_pickle("C:\Users\mmacfarlane\Desktop\MattSeven\Python\pickles\CombinedData.pkl")

##dfOptions['PnlToExp'] = dfOptions['PnlToExp'] * dfOptions['closePrice'] / 100
#dfOptions['Pnl2'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(10)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl2'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(2)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl4'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(4)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl6'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(6)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl8'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(8)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl10'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(10)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl12'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(12)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl14'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(14)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl16'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(16)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl18'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(18)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl20'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(20)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl22'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(22)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl24'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(24)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl26'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(26)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl28'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(28)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl30'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(30)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl32'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(32)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl34'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(34)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl36'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(36)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl38'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(38)) * 100 / dfOptions['closePrice']
#dfOptions['Pnl40'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(40)) * 100 / dfOptions['closePrice']
#
#dfOptions['2dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(2) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['4dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(4) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['6dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(6) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['8dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(8) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['10dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(10) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['12dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(12) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['14dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(14) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['16dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(16) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['18dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(18) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['20dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(20) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['22dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(22) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['24dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(24) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['26dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(26) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['28dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(28) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['30dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(30) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['32dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(32) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['34dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(34) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['36dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(36) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['38dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(38) - dfOptions['Mark']) * 100 / dfOptions['closePrice']
#dfOptions['40dPnlNH'] = (dfOptions['Mark'].groupby(level=['OptionID']).shift(40) - dfOptions['Mark']) * 100 / dfOptions['closePrice']

#
################################# Analysis #############################################
##dfOptions = pd.read_pickle("C:\Users\mmacfarlane\Desktop\MattSeven\Python\pickles\VData.pkl")
##print dfOptions
##dfOptions = pd.read_pickle("C:\Users\mmacfarlane\Desktop\MattSeven\Python\pickles\CombinedData1.pkl")
#
##dfTradeData = pd.read_pickle("C:\Users\mmacfarlane\Desktop\MattSeven\Python\pickles\TradeData.pkl").reset_index()
##dfTradeData['StockSymbol'] = dfTradeData['StockSymbol'].str.strip()
##dfTradeData = dfTradeData[dfTradeData['Trader'] == 'Matt']
##dfOptions = dfOptions.reset_index()
##dfOptions = dfOptions.merge(dfTradeData.reset_index(), how='inner', left_on=['Date','StockSymbol','OptionID'],right_on=['Date','StockSymbol','OptionID']).set_index(['Date','StockSymbol','OptionID'])
#
#dfOptions['ErnCount'] = np.where(dfOptions['DTErn']<dfOptions['DTExp'],1+np.floor((dfOptions['DTExp']-dfOptions['DTErn'])/90),0)
#dfOptions['XF'].fillna(value=0,inplace=True)
#dfOptions['XF20'] = np.sqrt(dfOptions['XF'] ** 2 * 126 * 3.14159 * dfOptions['ErnCount'] / dfOptions['DTExp'] + dfOptions['EMA20'] ** 2 * (dfOptions['DTExp'] - dfOptions['ErnCount']) / dfOptions['DTExp'])
#dfOptions['BuyEdgeXF20'] = dfOptions['XF20'] + dfOptions['SkewAdj'] - (dfOptions['ImpliedVolatility'])
##dfOptions['XFTony'] = np.sqrt(dfOptions['XF'] ** 2 * 126 * 3.14159 * dfOptions['ErnCount'] / dfOptions['DTExp'] + dfOptions['TonyVol'] ** 2 * (dfOptions['DTExp'] - dfOptions['ErnCount']) / dfOptions['DTExp'])
##dfOptions['BuyEdgeXFTony'] = dfOptions['XFTony'] + dfOptions['SkewAdj'] - (dfOptions['ImpliedVolatility'])
#dfOptions['absSD'] = abs(dfOptions['totalReturn'] / dfOptions['ImpliedVolatility'] * np.sqrt(252))
##dfOptions['ErnRatio'] = dfOptions['DTExp'] / dfOptions['ErnCount']
#
##dfOptions['XF120'] = np.sqrt(dfOptions['XF'] ** 2 * 126 * 3.14159 * dfOptions['ErnCount'] / dfOptions['DTExp'] + dfOptions['EMA120'] ** 2 * (dfOptions['DTExp'] - dfOptions['ErnCount']) / dfOptions['DTExp'])
##dfOptions['BuyEdgeXF120'] = dfOptions['XF120'] + dfOptions['SkewAdj'] - (dfOptions['IVA'] + dfOptions['IVB']) / 2
##dfOptions['BuyEdgeXF120Rank'] = dfOptions['BuyEdgeXF120'].groupby(level=['Date']).rank(pct=True)
##dfOptions['CenVol'] = np.sqrt((dfOptions['ImpliedVolatility'] ** 2 - dfOptions['XF'] ** 2 * 126 * 3.14159 * dfOptions['ErnCount'] / dfOptions['DTExp']) * (dfOptions['DTExp'] / (dfOptions['DTExp'] - dfOptions['ErnCount'])))
##dfOptions['VolRange'] = (dfOptions['CenVol'] - dfOptions['SkewAdj'] - dfOptions['minV']) / (dfOptions['maxV'] - dfOptions['minV'])
#
##dfOptions['FairVol'] = dfOptions['ImpliedVolatility'] - dfOptions['EMA20'] + dfOptions['10dPnl'] / 100 * dfOptions['closePrice'] / dfOptions['Vega']
##dfOptions['prevPnL'] = dfOptions['10dPnl'].groupby(level=['OptionID']).shift(-10)
##dfOptions['prevPnLRank'] = dfOptions['prevPnL'].groupby(level=['Date']).rank(pct=True)
#
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['0%Vol'])&(dfOptions['CenVol'] <= dfOptions['10%Vol']),'Perp'] =0+ (dfOptions['CenVol'] - dfOptions['0%Vol']) / (dfOptions['10%Vol'] - dfOptions['0%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['10%Vol'])&(dfOptions['CenVol'] <= dfOptions['20%Vol']),'Perp'] =0.1+ (dfOptions['CenVol'] - dfOptions['10%Vol']) / (dfOptions['20%Vol'] - dfOptions['10%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['20%Vol'])&(dfOptions['CenVol'] <= dfOptions['30%Vol']),'Perp'] =0.2+ (dfOptions['CenVol'] - dfOptions['20%Vol']) / (dfOptions['30%Vol'] - dfOptions['20%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['30%Vol'])&(dfOptions['CenVol'] <= dfOptions['40%Vol']),'Perp'] =0.3+ (dfOptions['CenVol'] - dfOptions['30%Vol']) / (dfOptions['40%Vol'] - dfOptions['30%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['40%Vol'])&(dfOptions['CenVol'] <= dfOptions['50%Vol']),'Perp'] =0.4+ (dfOptions['CenVol'] - dfOptions['40%Vol']) / (dfOptions['50%Vol'] - dfOptions['40%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['50%Vol'])&(dfOptions['CenVol'] <= dfOptions['60%Vol']),'Perp'] =0.5+ (dfOptions['CenVol'] - dfOptions['50%Vol']) / (dfOptions['60%Vol'] - dfOptions['50%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['60%Vol'])&(dfOptions['CenVol'] <= dfOptions['70%Vol']),'Perp'] =0.6+ (dfOptions['CenVol'] - dfOptions['60%Vol']) / (dfOptions['70%Vol'] - dfOptions['60%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['70%Vol'])&(dfOptions['CenVol'] <= dfOptions['80%Vol']),'Perp'] =0.7+ (dfOptions['CenVol'] - dfOptions['70%Vol']) / (dfOptions['80%Vol'] - dfOptions['70%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['80%Vol'])&(dfOptions['CenVol'] <= dfOptions['90%Vol']),'Perp'] =0.8+ (dfOptions['CenVol'] - dfOptions['80%Vol']) / (dfOptions['90%Vol'] - dfOptions['80%Vol']) * .1
##dfOptions.loc[(dfOptions['CenVol'] >= dfOptions['90%Vol'])&(dfOptions['CenVol'] <= dfOptions['100%Vol']),'Perp'] =0.9+ (dfOptions['CenVol'] - dfOptions['90%Vol']) / (dfOptions['100%Vol'] - dfOptions['90%Vol']) * .1
#
###dfOptions['NewXF'] = dfOptions['XF'] * np.where(dfOptions['DTErn']<=10,.5 + dfOptions['DTErn'] * .06,np.where(1.1 - dfOptions['DTErn'] * .015 < .5, .5, 1.1 - dfOptions['DTErn'] * .015 ))
###dfOptions['NewXF20'] = np.sqrt(dfOptions['NewXF'] ** 2 * 126 * 3.14159 * dfOptions['ErnCount'] / dfOptions['DTExp'] + dfOptions['EMA20'] ** 2 * (dfOptions['DTExp'] - dfOptions['ErnCount']) / dfOptions['DTExp'])
###dfOptions['NewBuyEdgeXF20'] = dfOptions['NewXF20'] + dfOptions['SkewAdj'] - dfOptions['IVA']
###dfOptions['NewBuyEdgeXF20Rank'] = dfOptions['NewBuyEdgeXF20'].groupby(level=['Date']).rank(pct=True)
#dfOptions['HasErn'] = np.where(dfOptions['DTExp']>dfOptions['DTErn'],1,0)
#
##dfOptions['BuyEdgeXF20Ern'] = dfOptions[dfOptions['HasErn']==1]['BuyEdgeXF20']
##dfOptions['BuyEdgeXF20RankErn'] = dfOptions['BuyEdgeXF20Ern'].groupby(level=['Date']).rank(pct=True)
##dfOptions['BuyEdgeXF20NoErn'] = dfOptions[dfOptions['HasErn']==0]['BuyEdgeXF20']
##dfOptions['BuyEdgeXF20RankNoErn'] = dfOptions['BuyEdgeXF20NoErn'].groupby(level=['Date']).rank(pct=True)
#
##dfOptions['10dPnl'] = dfOptions['10dPnl'] * dfOptions['closePrice'] / 100
##dfOptions['10dPnlNH'] = dfOptions['10dPnlNH'] * dfOptions['closePrice'] / 100
##dfOptions['10dGscalp'] = dfOptions['10dPnl'] - dfOptions['10dPnlNH']
#
#dfFiltered = dfOptions
#dfFiltered = dfFiltered[(dfFiltered['d']<=60) & (dfFiltered['d']>=40)]
##dfFiltered = dfFiltered[dfFiltered['quantity']>0] #Spelling error in pickle
##dfFiltered = dfFiltered[dfFiltered['Side']==1]
##dfFiltered = dfFiltered[(dfFiltered['Account'] == 'FNS') | (dfFiltered['Account'] == 'MNS')]
##dfFiltered = dfFiltered[(dfFiltered['d']>=80)]
#dfFiltered = dfFiltered[(dfFiltered['DTExp']>=21) & (dfFiltered['DTExp']<=42)]
##dfFiltered = dfFiltered[dfFiltered['AbsDelta']<.5]
##dfFiltered = dfFiltered[dfFiltered['Cost']<.30]
##dfFiltered = dfFiltered[(dfFiltered['BuyEdgeXFTonyRank']>.8) | (dfFiltered['BuyEdgeXFTonyRank']<.2)]
##dfFiltered = dfFiltered[dfFiltered['HasErn']==1]
##dfFiltered = dfFiltered[dfFiltered['10dIVchg']<98]
##dfFiltered = dfFiltered[dfFiltered['5dIVchg']<98]
##dfFiltered = dfFiltered[dfFiltered['1dIVchg']<98]
##dfFiltered = dfFiltered[dfFiltered['DTErn']>10]
##dfFiltered = dfFiltered[dfFiltered['DSErn']>10]
##dfFiltered = dfFiltered[dfFiltered['VolMoSRank']>.5]
##dfFiltered = dfFiltered[dfFiltered['isETF']==1]
##dfFiltered = dfFiltered[dfFiltered['CallPut']=='P']
##dfFiltered = dfFiltered[dfFiltered['XF']>.10]
##dfFiltered = dfFiltered[dfFiltered['XF']<.05]
##dfFiltered = dfFiltered[dfFiltered['closePrice']>100]
#
##dfFiltered = dfFiltered.reset_index()
##dfFiltered = dfFiltered[dfFiltered['StockSymbol'] == 'SPY']
##dfFiltered = dfFiltered[dfFiltered['Date']>'2008-01-01']
##dfFiltered = dfFiltered[dfFiltered['Date']<'2011-01-01']
##dfFiltered = dfFiltered.set_index(['Date','StockSymbol'])
#
########### Ranks #############
##dfFiltered['IVRank'] = dfFiltered['ImpliedVolatility'].groupby(level=['Date']).rank(pct=True)
#dfFiltered['BuyEdgeXF20Rank'] = dfFiltered['BuyEdgeXF20'].groupby(level=['Date']).rank(pct=True)
##dfFiltered['BuyEdgeXFTonyRank'] = dfFiltered['BuyEdgeXFTony'].groupby(level=['Date']).rank(pct=True)
##dfFiltered['PerpRank'] = dfFiltered['Perp'].groupby(level=['Date']).rank(pct=True)
##dfFiltered['VolRangeRank'] = dfFiltered['VolRange'].groupby(level=['Date']).rank(pct=True,ascending=False)
#dfFiltered['1dIVchgRank'] = dfFiltered['1dIVchg'].groupby(level=['Date']).rank(pct=True)
#dfFiltered['5dIVchgRank'] = dfFiltered['5dIVchg'].groupby(level=['Date']).rank(pct=True)
#dfFiltered['10dIVchgRank'] = dfFiltered['10dIVchg'].groupby(level=['Date']).rank(pct=True)
#dfFiltered['absSDRank'] = dfFiltered['absSD'].groupby(level=['Date']).rank(pct=True)
#
#dfFiltered.loc[dfFiltered['BuyEdgeXF20Rank']>.8,'Side'] = 1
#dfFiltered.loc[dfFiltered['BuyEdgeXF20Rank']<.2,'Side'] = -1
#dfFiltered = dfFiltered[(dfFiltered['Side']==1) | (dfFiltered['Side']==-1)]
#dfFiltered['5dPnlT'] = dfFiltered['5dPnl'] * dfFiltered['Side']
#dfFiltered['10dPnlT'] = dfFiltered['10dPnl'] * dfFiltered['Side']
#dfFiltered['21dPnlT'] = dfFiltered['21dPnl'] * dfFiltered['Side']
#dfFiltered['PnlToExpT'] = dfFiltered['PnlToExp'] * dfFiltered['Side']
#dfFiltered['5d Win%'] = np.where(dfFiltered['5dPnlT'] > 0,1,0)
#dfFiltered['10d Win%'] = np.where(dfFiltered['10dPnlT'] > 0,1,0)
#dfFiltered['21d Win%'] = np.where(dfFiltered['21dPnlT'] > 0,1,0)
#dfFiltered['ToExp Win%'] = np.where(dfFiltered['PnlToExpT'] > 0,1,0)
###Trade Data
#dfFiltered = dfFiltered[['5dPnlT','10dPnlT','21dPnlT','PnlToExpT','Side','CallPut','ImpliedVolatility','1dIVchg','5dIVchg','10dIVchg','DTExp','d','DTErn','DSErn','HV5','totalReturn','absSD','10dStock','10dHistStock','XF','EMA20','BuyEdgeXF20','quantity','IsOpen','isETF','HasErn','5d Win%','10d Win%','21d Win%','ToExp Win%']]
##Test Data
##dfFiltered = dfFiltered[['5dPnlT','10dPnlT','21dPnlT','PnlToExpT','Side','CallPut','ImpliedVolatility','1dIVchg','5dIVchg','10dIVchg','1dIVchgRank','5dIVchgRank','10dIVchgRank','DTExp','d','DTErn','DSErn','HV5','10dStock','10dHistStock','EMA20','XF','XF20','BuyEdgeXF20','BuyEdgeXF20Rank','isETF','HasErn','5d Win%','10d Win%','21d Win%','ToExp Win%']]
#
#dfFiltered.reset_index().to_excel("C:\Users\mmacfarlane\Desktop\MattSeven\Python\data\dfTonyTrades.xlsx")

############# Factor Analysis ######################
#xCol = 'BuyEdgeXF20Rank'
#yCol = '10dPnl'
##print dfOptions.count()
#
#xQmin = .02
#xQmax = .98
#yQmin = .2
#yQmax = .8

############### Graph PnL ########################
#longs = dfFiltered[yCol].copy()
#longs[:] = np.nan
#longs[(dfFiltered[xCol]>0)] = 1
#longpnl = dfFiltered.loc[longs == 1,yCol].mean(level=['Date','StockSymbol']).mean(level='Date')
#longpnltrades = dfFiltered.loc[longs == 1,yCol].mean(level=['Date','StockSymbol']).count(level='Date')
#longpnl.plot()
#print longpnl.mean(), longpnltrades.mean()
##print ((longpnl.dropna() / 100 + 1)).prod()

#shorts = dfFiltered[yCol].copy()
#shorts[:] = np.nan
#shorts[(dfFiltered[xCol]<100) & (dfFiltered[xCol]>0)] = 1
#shortpnl = dfOptions.loc[shorts == 1,yCol].mean(level=['Date','StockSymbol']).mean(level='Date')
#shortpnl.plot()
#print shortpnl.mean()
#print ((shortpnl.dropna() / 100 + 1)).prod()
#
#pnl = longpnl - shortpnl
##pnl = dfFiltered.loc[longs == 1,yCol].mean(level=['Date','StockSymbol']).mean(level='Date') - dfFiltered.loc[shorts == 1,yCol].mean(level=['Date','StockSymbol']).mean(level='Date')
#pnl.plot()

#plt.show()
#print pnl.mean()

############### Polyfit################
#dfPoly = dfFiltered[[xCol,yCol]].dropna()
#dfPoly = dfPoly[(dfPoly[xCol]>dfPoly[xCol].quantile(xQmin)) & (dfPoly[xCol]<dfPoly[xCol].quantile(xQmax))]
###dfOptions = dfOptions[[xCol,yCol]].mean(level=['Date','StockSymbol']).dropna()
#print dfPoly.count()
#
##dfOptions.plot.hexbin(x=xCol,y=yCol,gridsize=20,mincnt=0,extent=[dfOptions[xCol].quantile(xQmin),dfOptions[xCol].quantile(xQmax),dfOptions[yCol].quantile(yQmin),dfOptions[yCol].quantile(yQmax)])
#
#degrees = 4
#polyWeights = np.polyfit(dfPoly[xCol],dfPoly[yCol],degrees)
#fit = 0.0
#polyy = 0.0
#n = 0
#polyx = np.arange(dfPoly[xCol].quantile(xQmin), dfPoly[xCol].quantile(xQmax), (dfPoly[xCol].quantile(xQmax) - dfPoly[xCol].quantile(xQmin)) / 100)
#for term in polyWeights:
#    fit = fit + polyWeights[n] * dfPoly[xCol] ** (degrees - n)
#    polyy = polyy + polyWeights[n] * polyx ** (degrees - n)
#    n = n + 1
#error = abs(fit - dfPoly[yCol])
#print polyWeights, error.mean()
#plt.plot(polyx, polyy)
#
#print dfPoly.loc[dfPoly[xCol]>=dfPoly[xCol].quantile(0),yCol].mean()

##################################Decision Tree###########################################
#dfOptions = dfOptions.dropna()
#treeCols = ['VolMoSRank','IVWidth', 'StockRange', 'DTExp', 'DTErn', 'ImpliedVolatility','OpenInterestRatio','VIX Level','BuyEdgeXF20Rank','HasErn','5dIVchgRank','10dIVchgRank', 'prevPnL']
##treeCols = ['VolMoSRank','IVWidth', 'DTExp', 'DTErn','BuyEdgeXF120Rank','BuyEdgeXF20Rank','5dIVchgRank','10dIVchgRank']
#
##model = RandomForestClassifier()
#
##model = tree.DecisionTreeRegressor(max_depth=4)
##classify = model.fit(dfOptions.as_matrix(columns=treeCols),dfOptions.as_matrix(columns=['21dPnl']))
#
#model = tree.DecisionTreeClassifier(max_depth=5,min_samples_split=dfOptions['10dPnl'].count() / 100) #,class_weight="balanced"
#classify = model.fit(dfOptions.as_matrix(columns=treeCols),dfOptions.as_matrix(columns=['10dPnlPos']))
#
##tree.export_graphviz(classify,out_file='dt.dot')
##command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
##subprocess.check_call(command)
#dfOptions['treePredict'] = classify.predict(dfOptions.as_matrix(columns=treeCols))
##dfOptions['treePredict']=results
#
##print classify.get_params()
#print dfOptions.loc[dfOptions['treePredict']>0,'21dPnl'].mean()
#print round(dfOptions.loc[dfOptions['treePredict']>0,'21dPnl'].count() * 100.0 / dfOptions['21dPnl'].count(),2), "trade%"
#print round(dfOptions.loc[dfOptions['treePredict']>0,'21dPnlPos'].sum() * 100.0 / dfOptions.loc[dfOptions['treePredict']>0,'21dPnlPos'].count(),2), "win%"
##with open('tree.dot', 'w') as dotfile:
##    tree.export_graphviz(
##        classify,
##        dotfile,
##        feature_names=treeCols)
###    dot -Tpng tree.dot -o tree.png
#
#dot_data = StringIO() 
#tree.export_graphviz(classify, out_file=dot_data, feature_names=treeCols,filled=True,proportion=True)
#graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
#graph.write_pdf("tree.pdf")
#Image(graph.create_png())
#
#print treeCols
#print model.feature_importances_


########################## Time ############################
print round(time.time() - startTime, 2), "seconds"
