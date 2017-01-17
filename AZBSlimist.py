# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 20:42:40 2017

@author: jary
"""

Created on Tue Jan 20 15:48:35 2015
 
@author: azhang

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
 
 
pd.options.mode.chained_assignment = None  # default='warn'
 
 
import AZ_CVs
 
class BS:
    '''Calculate 63 day fair cv and BSlimit.'''
   
    def __init__(self, CV, NGinfo, WeightPara1, SMR_Weight, BSAdj):      
        self.CV = CV
        self.NGinfo = NGinfo
        self.WeightPara1 = WeightPara1
        self.SMR_Weight = SMR_Weight
        self.BSAdj = BSAdj
       
        
    def fv_calculator(self):
        '''Calculate fair cv using cv in different terms.'''
        FVar_temp = self.CV['c_v_10'] ** 2 * self.WeightPara1[0] / 100 \
        + self.CV['c_v_21'] ** 2 * self.WeightPara1[1] / 100 \
        + self.CV['c_v_63'] ** 2 * self.WeightPara1[2] / 100 \
        + self.CV['c_v_126'] ** 2 * self.WeightPara1[3] / 100 \
        + self.CV['c_v_252'] ** 2 * self.WeightPara1[4] / 100
        FV_temp = FVar_temp.apply(np.sqrt)
        Matrix = self.CV
        Matrix['FV_temp'] = FV_temp
        #adjust for censored days:
        #Matrix['FV'] = Matrix['FV_temp'] * (Matrix['c_days'] / 100 * 2 + 1)
        Matrix['FV'] = Matrix[['FV_temp', 'c_days']].apply(lambda srs: srs[0] * (max(srs[1]-1,0) / 10 * 0.5 + 1) + max(srs[1]-1,0) * 0.5, axis=1)
        return Matrix
 
 
    def fv_adjust(self):
        '''Calculate adjusted fair.'''
        Matrix = self.fv_calculator() 
        
        Matrix.Ticker = map(lambda x: x.replace(" ",""), Matrix.Ticker) #trim spaces
        Matrix = pd.merge(Matrix, self.NGinfo, on='Ticker', sort=False)
        #Matrix = Matrix.join(self.NGinfo, on = 'Ticker', how='left', lsuffix='', rsuffix='_dup', sort=False)
       
        def adj_SMR10(fv, SMR10):
            '''
            Adjust with 2-year-period SMRdampen of 10 day cv.
            '''
            result = fv * (1 - self.SMR_Weight) + SMR10 * self.SMR_Weight
            return result
       
        def adj_absvol(fv):
            '''
            Adjust for absolute vol.
            Adjustment is max[0.95, 1-(|FV-25|)/(FV-25)*sqrt(|FV-25|)/40]
            FV=15 => FV_adj_absvol=16.16; FV=40 => FV_adj_absvol=38;
            '''
            result = fv * (1 - (fv - 25.001).abs() / (fv - 25.001) * (fv - 25.001).abs().apply(np.sqrt) / 40).apply(lambda x: max(x, 0.950))
            return result           
 
        def adj_mktcap(fv, mktcap):
            '''
            Adjust for mkt cap.
            Adjustment is FV*(0.85+1/ln(mktcap))
            mktcap=1000 => adj_absvol=1.067; mktcap=200,000 => adj_absvol=0.97;
            '''
            result = fv * (0.85 + 1.5 / mktcap.apply(np.log))
            return result
       
#        def adj_RR(fv, StockRiskRank, StockRiskCategory):
#            '''
#            Adjust for Risk Rank
#            '''
#            result = fv * (1 + StockRiskRank / 300)
#            return result
        Matrix['FV_adj_SMR10'] = adj_SMR10(Matrix['FV'], Matrix['SMR10'])
        Matrix['FV_adj_absvol'] = adj_absvol(Matrix['FV_adj_SMR10'])
        Matrix['FV_adj_absvol_mktcap'] = adj_mktcap(Matrix['FV_adj_absvol'], Matrix['MarketCap'])
#        Matrix['FV_adj_absvol_mktcap_RR'] = adj_RR(Matrix['FV_adj_absvol_mktcap'], Matrix['StockRiskRank'], Matrix['StockRiskCategory'])
        return Matrix
       
    def bs_limit(self):
        '''Calculate BSlimits.'''
        Matrix = self.fv_adjust()
       
        #Adjust for LiqRank
        Matrix['LiqAdj'] = 0.0
        Matrix['LiqAdj'][Matrix['LiqRank'] > 1500] = 0.06
        Matrix['LiqAdj'][(Matrix['LiqRank'] <= 1500) & (Matrix['LiqRank'] >= 900)] = 0.02
       
        #Adjust for MktCap
        Matrix['MktCapAdj_Sell'] = 0.0
        Matrix['MktCapAdj_Sell'][Matrix['MarketCap'] < 300] = 0.09
        Matrix['MktCapAdj_Sell'][(Matrix['MarketCap'] >= 300) & (Matrix['MarketCap'] < 1000)] = 0.07
        Matrix['MktCapAdj_Sell'][(Matrix['MarketCap'] >= 1000) & (Matrix['MarketCap'] < 2500)] = 0.02
        Matrix['MktCapAdj_Sell'][(Matrix['MarketCap'] <= 100000) & (Matrix['MarketCap'] > 30000)] = -0.025
        Matrix['MktCapAdj_Sell'][Matrix['MarketCap'] > 100000] = -0.05
       
        #Adjust for industry
        Matrix['IndustryAdj'] = 0.0
       
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("BANKKS")] = 0.25
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("REALTE")] = 0.25
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("SPCFIN")] = 0
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("FINSVC")] = 0
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("INSURE")] = 0
       
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("RETDIS")] = 0.5
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("APPTXT")] = 0.5
       
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("BIOPHM")] = 1
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("MEDIEQ")] = 0
        Matrix['IndustryAdj'][Matrix['Industry'].str.contains("HLTHSV")] = 0.5
       
        Matrix['Blimits'] = Matrix['FV_adj_absvol_mktcap'] * (1 - self.BSAdj[0] - Matrix['LiqAdj']) - self.BSAdj[1] + Matrix['IndustryAdj']
        Matrix['Slimits'] = Matrix['FV_adj_absvol_mktcap'] * (1 + self.BSAdj[0] + Matrix['LiqAdj'] + Matrix['MktCapAdj_Sell']) + self.BSAdj[1] + Matrix['IndustryAdj']
       
        #Matrix = Matrix.drop(labels = 'Ticker_dup',axis=1)
       
        return Matrix
       
    def export(self):
        '''Choose column needed to export to csv'''
        Matrix = self.bs_limit()
        Output = Matrix[['Ticker', 'Industry', 'FV_adj_absvol_mktcap', 'Blimits', 'Slimits', 'MarketCap']]
        return Output
       
if __name__ == '__main__':
    CV_class = AZ_CVs.CVs()
    CV = CV_class.GroupBy()
    NGinfo = CV_class.get_tickers()
    BS_class = BS(CV, NGinfo, [10, 10, 25, 30, 25], 0.25, [0.04, 0.85])
    '''Change [10, 21, 63, 126, 252] days weighting; and change SMR10 weighting ; and change [BS half width percentage and points, eg. Slimit = fv*(1+0.05)+0.85]'''
    '''Default: [5, 5, 20, 40, 30], 0.2, [0.09, 0.75]'''
    #check_fv = BS_class.fv_calculator()
    #check_fvAllInfo = BS_class.fv_adjust()
    BSAllInfo = BS_class.bs_limit()
   
    print "************ BS_limits Done ************"