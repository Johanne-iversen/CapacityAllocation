from BackBone import config as cfg
from BackBone.preprocessing import loader
from pathlib import Path, PosixPath
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import itertools
import numpy as np
import os


def model_validation(var):
    x_y_validation(var)
    SS_v_validation(var)
    v_q_validation(var)
    capacity_validation(var)
    demand_validation(var)

    return


# check if y and x always aligns
def x_y_validation(variables):
    '''
    Checking if y == 1 if x > 0 and vice versa
    '''
    check = variables[(variables['x']<= 0) & (variables['y'] > 0.1)]
    if len(check) == 0:
        print('x and y constraints are satisfied')
    else:
        print(len(check), ' rows violates the x and y condition')


# check if s < 1 if SS is not met
def SS_v_validation(variables):
    try: 
        cp = pd.read_pickle(cfg.CUST_PROD_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))
    
    cust_prod_df = pd.DataFrame(cp).T.reset_index()
    cust_prod_df.columns = ['Customer','Product_id','inv','beta','SS']
    variables = variables.merge(cust_prod_df, how='left', on =['Customer','Product_id'])

    check = variables[(variables['SS'] > variables['v']+0.1) & (variables['s']==1)]
    if len(check) == 0:
        print('service level according to safety stock is satisfied')
    else:
        print(len(check), ' rows violates the service level condition')
    return check


# check if q = 0 if v > 0 and vice versa
def v_q_validation(variables):
    check = variables[(variables['v'] >0.1) & (variables['q'] > 0.1)]

    if len(check) == 0:
        print('v and q relation is satisfied')
    else:
         print(len(check), ' rows violates the v and q condition')
    return check



# chekcing if capacity is not exeeded
def capacity_validation(variables):
    try:
        Site = pd.read_pickle(cfg.SITE_PATH)
        pv = pd.read_pickle(cfg.PRODUCT_VARIANT_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))
    
    site = pd.DataFrame(Site).T.reset_index()
    site.columns=['Site','Period','cap','co','mb', 'u']
    site = site.drop(['mb'],axis=1)
    site['Period'] = site['Period'].astype(str)

    variables = variables.groupby(['Site','Period','iteration']).sum()[['x','y']]
    
    variables = variables.merge(site, how='left', on =['Site','Period'])

    variables['used_cap'] = variables['x'] + variables['y']*variables['u']*variables['co']

    check = variables[(variables['used_cap'] > variables['cap']+0.1)]
    if len(check) == 0:
        print('the capacity constraint is satisfied')
    else:
        print(len(check), ' rows violates the capacity condition')
    return check

# cheking if demand is met or if buffer catches unmet demand
def demand_validation(variables):
    try:
        pv = pd.read_pickle(cfg.PRODUCT_VARIANT_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))

    demand = pd.DataFrame(pv).T.reset_index()
    demand.columns=['Site', 'Customer', 'type', 'Product_id', 'Period', 'Demand']
    demand['Period'] = demand['Period'].astype(str)

    variables = variables.merge(demand, how='left', on=['Site','Customer', 'Product_id', 'Period'])

    variables = variables.groupby(['Site','Customer','Product_id','iteration']).sum()[['x','d','Demand']].reset_index()

    check = variables[variables['x'] + variables['d'] +0.1 < variables['Demand']]
    if len(check) == 0:
        print('The demand constraint is satisfied')
    else:
        print(len(check), ' rows violates the demand condition')
    return check
