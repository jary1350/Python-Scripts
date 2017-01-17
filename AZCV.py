# -*- coding: utf-8 -*-
"""
Created on Mon Dec 08 17:37:49 2014
 
@author: azhang
"""
 
from beef import sql_script_runner
import datetime
import os
print os.getpid()
import pymmd
from collections import defaultdict
import socket
import os
import pandas.io.sql as pdsql
import pyodbc #sql
from pyspark import pyspark as ps
import numpy as np #math
import pandas as pd #dataframe
import time
print os.getpid()
import math
import smtplib
#user = (socket.gethostname())
 
 
 
##Send Email
#def sendEmail(ScriptName, ResultMessage):
#    smtpObj = smtplib.SMTP('scwchi5pcas1.peak6.net')
#    message = '''Subject: ''' + ScriptName + ''': ''' + ResultMessage + '''
#    '''
#    smtpObj.sendmail('mruiz@peak6.com', 'mruiz@peak6.com', message)
#    smtpObj.sendmail('mruiz@peak6.com', 'mruiz@peak6.com', message)
 
class CVs:
    '''Calculate Censored Vol using historical data.'''
   
    def __init__(self):
        ps.init()
 
 
    def get_tickers(self):
        '''Returns a dataframe of tickers and its related infomation in the universe.'''
        NGquery = '''(And
                
                 (In Subindustries "SectorGroup2")                  
                 
                 (Set HistIV (HistVol 275 "IV_63"))
                 (IsValid HistIV)
               (HasOptions)
               (Set! Ticker StockSymbol)
                 (ExportValues Industry MarketCap StockRiskRank StockRiskCategory)
               (Set! LiqRank (DMlong "DM_Locasto_LiquidityRank" StockSymbol "DM_Risk_LiqudityRank"))
            
                 (Set! SMR10 (SMRDampen (HistVolRange 500 1 "C_DM_10")))
                
                )'''
        return ps.stock_script(NGquery).make_dataframe()
   
    
    def ticker_str(self):
        '''Return ..or..or.. str that is used in SQL script.'''
        ticker = self.get_tickers()
        return "s.Ticker = '" + "' or s.Ticker = '".join(ticker['Ticker']) + "'"
 
 
    def get_SQL_data1(self):
        '''Returns a dataframe of tickers and 252 days of daily return data.'''
        SQL_str = self.ticker_str()
       
        SQLquery =r"""
         SELECT
               CONVERT(date,sp.[Date]) AS 'DATE'
              ,sp.[TotalReturn]
              ,s.Ticker
              ,isnull(e.DateType, 'NA') as Earning
              ,c.TradeCardinal
          FROM [IvyDB].[dbo].[SECURITY_PRICE] sp
          LEFT JOIN [IvyDB].[dbo].[SECURITY] s
          ON sp.SecurityID = s.SecurityID
          LEFT JOIN [CMSandbox].[dbo].[tblCalendar] c
          ON sp.[DATE] = c.CalendarDate
          LEFT JOIN [companies].[dbo].[tblStocks] i
          ON s.Ticker = i.StockSymbol
          LEFT JOIN (select * from [companies].[dbo].[tblEarnings] where IsDeleted = 0) e
          ON i.CompanyID = e.CompanyID
          AND CONVERT(date,sp.[Date]) = CONVERT(date,e.[Date])
          where
          c.TradeCardinal >= (select AdjTradeCardinal from [CMSandbox].[dbo].[tblCalendar] where CalendarDate = CONVERT(date, GETDATE())) - 300
          and (s.Class = '')        
          and (%s)         
          --and (s.Ticker = 'BRK.B' or s.Ticker = 'AGO')  
          order by ticker, TradeCardinal        
        """ % (SQL_str)
        df_raw = sql_script_runner(SQLquery).get_df()
 
        #df_raw["Ticker"] = df_raw["Ticker"].map(str.strip)
        #df_raw["Ticker"] = df_raw["Ticker"].strip()
        return df_raw
       
        
    def get_demean_vol(self,l):
        '''Calculate demeaned std for a given list.'''
        l = l.add(-np.mean(l))
        l_sqr = [ele ** 2 for ele in l]
        try:
            demean_v = math.sqrt(sum(l_sqr) / len(l_sqr)) * 100
        except ZeroDivisionError:
            demean_v = 999
        return demean_v
   
    
    def censor_earning(self, df_raw):
        '''Censor earning and selected dates' data from the whole dataframe and make it 252 days data.'''
        cardinal_list_temp = list(df_raw['TradeCardinal'][df_raw['Earning'] != 'NA'])
       
        cardinal_list = cardinal_list_temp + [e-1 for e in cardinal_list_temp] + \
        [e+1 for e in cardinal_list_temp] + \
        [e+2 for e in cardinal_list_temp]
       
        # Add my date (TradeCardinal) here to censor out: [xxxx, yyyy, zzzz]
        mydate_list = [4744, 4745]
        censor_list = list(set(cardinal_list + mydate_list))
       
        rtn_without_e_without_mydate = df_raw[~df_raw['TradeCardinal'].isin(censor_list)]
       
        df_without_e_without_mydate = rtn_without_e_without_mydate[len(rtn_without_e_without_mydate)-252:]
        #print df_without_e_without_mydate
        return df_without_e_without_mydate
       
        
        
    
    def cv_calculator(self, df_raw):
        '''Calculate censored vol using daily return data.'''
        df = self.censor_earning(df_raw)
        log_rtn = np.log(df[['TotalReturn']].add(1))
        raw_v_252 = self.get_demean_vol(log_rtn['TotalReturn']) * math.sqrt(252)
       
        censor_std_threshold = 4.5
        censor_threshold = censor_std_threshold * raw_v_252 / 100 / 16   
        
        def cv_calculator_sub(days):
            '''Calculate n-days cv.'''
            #c_rtn = log_rtn[(252-days):][abs(log_rtn[(252-days):]['TotalReturn'])<censor_threshold]
            c_rtn = log_rtn[(252-days):][abs(log_rtn[(252-days):]['TotalReturn'])<censor_threshold]
            c_v = self.get_demean_vol(c_rtn['TotalReturn']) * math.sqrt(252)
            return c_v, len(c_rtn), c_rtn
   
        c_v_252 = cv_calculator_sub(252)[0]
        c_v_126 = cv_calculator_sub(126)[0]
        c_v_63 = cv_calculator_sub(63)[0]
        c_v_42 = cv_calculator_sub(42)[0]
        c_v_21 = cv_calculator_sub(21)[0]
        c_v_10 = cv_calculator_sub(10)[0]
        c_days = 252 - cv_calculator_sub(252)[1]
   
        #print cv_calculator_sub(252)[2].to_string()
   
        tempdata = [{#'Ticker':df['Ticker'][0], \
        'c_v_252':c_v_252, \
        'c_v_126':c_v_126, \
        'c_v_63':c_v_63, \
        'c_v_42':c_v_42, \
        'c_v_21':c_v_21, \
        'c_v_10':c_v_10, \
        'c_days':c_days, \
        'c_threshold' : censor_threshold
        }]
 
        cols=['c_v_10','c_v_21','c_v_42','c_v_63','c_v_126','c_v_252','c_days', 'c_threshold']
       
        
        return pd.DataFrame(tempdata)[cols]
    
     
    def GroupBy(self):
        df = self.get_SQL_data1()
        temp = df.groupby(['Ticker']).apply(self.cv_calculator)
        return temp.reset_index(inplace=False).drop(labels = 'level_1',axis=1) #, inplace=False
       
        
    
if __name__ == '__main__':
 
    CV_class = CVs()
    CV_matrix = CV_class.GroupBy()
    CV_NGdata = CV_class.get_tickers()
    #X = CV_class.get_SQL_data1()

