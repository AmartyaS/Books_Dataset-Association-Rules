# -*- coding: utf-8 -*-
"""
Created on Sat Apr 10 03:06:37 2021

@author: Amartya
"""


import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder

data=pd.read_csv("F:\Softwares\Data Science Assignments\Python-Assignment\Association Rules\\book.csv")
data.head()
data.isna().sum()

#Formation of Association Rules
frequent_itemsets=apriori(data,min_support=0.1,use_colnames=True) 
#if minimum support is increased then frequent item set gets reduced thus, leading to lesser rules.
rules=association_rules(frequent_itemsets,metric="lift",min_threshold=0.8)

#Now we need to eliminate duplicate rules 
def tolist(i):
    return (sorted(list(i)))

maxrul=rules.antecedents.apply(tolist) +rules.consequents.apply(tolist)
maxrul=maxrul.apply(sorted)

rules_sets=list(maxrul)

uniquer=[list(m) for m in set(tuple(i) for i in rules_sets)]

indexr=[]
for i in uniquer:
    indexr.append(rules_sets.index(i))

unique_rules=rules.iloc[indexr,:]
