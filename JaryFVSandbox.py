# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 15:08:06 2017

@author: jary

"""

import pandas as pd
import numpy as np

FilePath = r'C:\Users\jary\Downloads\DMExample.pkl'
DataFrame = pd.read_pickle(FilePath)
Symbols = DataFrame.columns

class vols_df(pd.DataFrame):
    
    @classmethod
    def from_NG_dm(cls,NG_dm,vol_type):
        s1 = NG_dm.loc[vol_type]
        df1 = pd.DataFrame()
        for v in (s1.index):
            df1[v]=s1[v]
        df1.index.name = 'Date'
        return df1

IV = vols_df.from_NG_dm(DataFrame,'IV-84')






    

    

    

    





