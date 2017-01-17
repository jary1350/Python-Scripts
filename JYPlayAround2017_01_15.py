# -*- coding: utf-8 -*-
"""
Created on Sun Jan 15 09:52:23 2017

@author: jary
"""
import pandas as pd
import numpy as np

OptionID=['a','b','c','d','e']
dfOptions=pd.DataFrame({'PnlToExp':list(range(100,90,-2)),'OptionID':OptionID,'closePrice':list(range(100,90,-2)),\
                        'DaysToExp':[5,5,5,5,5]})
dfOptions2=pd.DataFrame({'PnlToExp':list(range(90,80,-2)),'OptionID':OptionID,'closePrice':list(range(90,80,-2)),\
                         'DaysToExp':[4,4,4,4,4]})
dfOptions=dfOptions.append(dfOptions2,ignore_index = True)

dfOptions2=pd.DataFrame({'PnlToExp':list(range(80,70,-2)),'OptionID':OptionID,'closePrice':list(range(80,70,-2)),\
                         'DaysToExp':[3,3,3,3,3]})

dfOptions=dfOptions.append(dfOptions2,ignore_index = True)


#dfOptions['Pnl2'] = (dfOptions['PnlToExp'] - dfOptions['PnlToExp'].groupby(level=['OptionID']).shift(2)) * 100 / dfOptions['closePrice']
#
