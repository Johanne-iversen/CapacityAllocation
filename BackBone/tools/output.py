from BackBone import config as cfg
from pathlib import Path, PosixPath
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import os
import itertools
import numpy as np
from pandas import ExcelWriter
from datetime import datetime
import math



################# Getting relevant decision varianble ###################
def getVars(decision_var_name, dict):
    '''
    Findes relevant decision variables from results dictonary
    '''
    updated_results = []
    result_dict = {}

    for key in dict.keys():
        if key[0] == decision_var_name:
            key_new = list(key)
            key_len = len(key_new)
            key_new.append(dict[key])
            updated_results.append(key_new)

        result_dict = {tuple(var[:key_len]): var[key_len] for var in updated_results}
    return updated_results

################## Getting objective values #####################
def solution_PM(var): 
    ''' 
    This function computes the objective values of a solution
    '''
    try:
        cust_prod_type_dict = pd.read_pickle(cfg.CUSTOMER_PRODUCT_TYPE_PATH)
        #cust_prod_dict = pd.read_pickle(cfg.CUST_PROD_PATH)
        data = pd.read_pickle(cfg.DATA_PICKLE_PATH)
        # cp_comb, Inv, P, Bias = gp.multidict(cust_prod_dict)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))

    cust_prod_df = pd.DataFrame(cust_prod_type_dict).T.reset_index()
    cust_prod_df.columns = ['Customer','type','Product_id','profit']
    x_var = pd.DataFrame(getVars('x',var), columns = ['Var','Site','Customer','type','Product_id','Period','Value'])
    
    x_var = x_var.groupby(['Customer','Product_id'])['Value'].sum().reset_index()

    revenue = x_var.merge(cust_prod_df, how='left')
    revenue['revenue'] = revenue['profit']*revenue['Value']

    tot_revenue = revenue.sum()['revenue']

    s_var = pd.DataFrame(getVars('s',var), columns = ['Var','Customer','Prodict_id','Period','Value'])
    sl_var = pd.DataFrame(getVars('sl',var), columns = ['Var','Customer','Prodict_id','Period','Value'])
    
    tot_SL = sl_var.sum()['Value']
    tot_S = s_var.sum()['Value']

    return tot_S, tot_SL, tot_revenue

################## Export decision variables #####################
def exportDecisionVariables(vars, i):
    '''
    This function extracts the decision variables and their values of each iteration in 
    the epsilon optimisation, marking the iteration and scenario of the corresponding solution
    '''

    #initialising with x
    variables= pd.DataFrame(getVars('x',vars),  columns = ['Var','Site','Customer','type','Product_id','Period','x']).drop('Var', axis=1)
    variables['iteration'] = i

    # loading remaining decision variables
    y_vars = pd.DataFrame(getVars('y',vars), columns = ['Var','Site','Customer','type','Product_id','Period','y']).drop('Var', axis=1)
    v_vars = pd.DataFrame(getVars('v',vars), columns = ['Var','Customer','Product_id','Period','v']).drop('Var', axis=1)
    q_vars = pd.DataFrame(getVars('q',vars), columns = ['Var','Customer','Product_id','Period','q']).drop('Var', axis=1)
    s_vars = pd.DataFrame(getVars('s',vars), columns = ['Var','Customer','Product_id','Period','s']).drop('Var', axis=1)
    sl_vars = pd.DataFrame(getVars('sl',vars), columns = ['Var','Customer','Product_id','Period','sl']).drop('Var', axis=1)
    d_vars = pd.DataFrame(getVars('d',vars), columns = ['Var','Site','Customer','Product_id','d']).drop('Var', axis=1)
    d_vars['Period'] = variables.min()['Period']
    z_vars = pd.DataFrame(getVars('z',vars), columns= ['Var', 'Customer', 'Product_id', 'Period', 'z']).drop('Var', axis=1)

    #join dataframes 
    variables = variables.merge(y_vars, how='left', on=['Site','Customer','type','Product_id','Period'])
    variables = variables.merge(v_vars, how='left', on=['Customer','Product_id','Period'])
    variables = variables.merge(q_vars, how='left', on=['Customer','Product_id','Period'])
    variables = variables.merge(s_vars, how='left', on=['Customer','Product_id','Period'])
    variables = variables.merge(sl_vars, how='left', on=['Customer','Product_id','Period'])
    variables = variables.merge(d_vars, how='left', on=['Site','Customer','Product_id','Period'])
    variables = variables.merge(z_vars, how='left', on=['Customer','Product_id','Period'])

    return variables


################# identifying pareto optimal solutions #########################
def paretooptimal(results):
    ''' 
    creating a dataframe and identifying pruned solutions
    '''
    solution = pd.DataFrame(results)
    solution.columns = ['S', 'SL', 'Revenue']
    solution['SL_total'] = solution['S']-solution['SL']
    solution['iteration'] = solution.index +1


    pareto_frontier = []
    for i in range(len(solution)):
        check = 0
        for j in range(len(solution)):
            if j!=i:
                if solution['SL_total'][j]>=solution['SL_total'][i] and solution['Revenue'][j]>=solution['Revenue'][i]:
                    check = 1
        # if solution['optimal'][i] == 1:
        #     check = 2
        pareto_frontier.append(check)
    solution['Paretofront'] = pareto_frontier

    max_S = solution['SL_total'].max()
    max_Revenue = solution['Revenue'].max()
    min_S = solution[solution['Paretofront']==0].min()['SL_total']
    min_Revenue = solution[solution['Paretofront']==0].min()['Revenue']

    euclidean = []
    optimal = []
    for i in range(len(solution)):
        if solution['Paretofront'][i] == 1:
            dist = 1
        else:
            dist_s = math.sqrt((solution['SL_total'][i]-max_S)**2) / (math.sqrt((solution['SL_total'][i]-max_S)**2) + math.sqrt((solution['SL_total'][i]-min_S)**2))
            dist_r = math.sqrt((solution['Revenue'][i]-max_Revenue)**2) / (math.sqrt((solution['Revenue'][i]-max_Revenue)**2) + math.sqrt((solution['Revenue'][i]-min_Revenue)**2))
            dist = dist_r + dist_s
        euclidean.append(dist)

    solution['euclidean'] = euclidean
    solution['optimal'] = 0
    t = [i for i in range(len(euclidean)) if euclidean[i] == min(euclidean)]
    solution['optimal'][t[0]] = 1
    solution['Paretofront'][t[0]] = 2
    
    return solution



# ####################### OUTPUT TO EXCEL #################################

def excel_output(variables, results, scenario):
    '''
    Exporting all decision variables, objective values and scenario specific input data to excel
    Input data is included, to ensure correct capacity, inventory and forecast when evaluating the scenario
    '''
    try:
        cp = pd.read_pickle(cfg.CUST_PROD_PATH)
        Site = pd.read_pickle(cfg.SITE_PATH)
        pv = pd.read_pickle(cfg.PRODUCT_VARIANT_PATH)
        cpt = pd.read_pickle(cfg.CUST_PROD_TIME_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))
    
    cp = pd.DataFrame(cp).T.reset_index()
    cp.columns=['Customer', 'Product_id', 'Inv',  'Bias', 'SS']
    cp = cp.drop('Bias', axis=1)

    site = pd.DataFrame(Site).T.reset_index()
    site.columns=['site','Period','cap','co','mb', 'u']
    site = site.drop(['site','co','mb', 'u'],axis=1)
    site['Period'] = site['Period'].astype(str)

    demand = pd.DataFrame(pv).T.reset_index()
    demand.columns=['Site', 'Customer', 'type', 'Product_id', 'Period', 'Demand']
    demand['Period'] = demand['Period'].astype(str)
    demand['type'] = demand['type'].astype(str)

    cpt = pd.DataFrame(cpt).T.reset_index()
    cpt.columns=['Customer', 'Product_id','Period','GSF']
    cpt['Period'] = cpt['Period'].astype(str)

    variables = variables.merge(cp, how='left', on=['Customer', 'Product_id'])
    variables = variables.merge(cpt, how='left', on =['Customer', 'Product_id','Period'])
    variables = variables.merge(demand, how = 'left', on =['Site', 'Customer', 'type', 'Product_id', 'Period'])

    minweek = min(variables['Period'])
    inv = []
    for i in range(len(variables)):
        if variables['Period'][i] == minweek:
            # variables['Inv'][i] = 0
            inv.append(variables['Inv'][i])
        else:
            inv.append(0)
    variables['Inv'] = inv
    variables['scenario'] = scenario
    results['scenario'] = scenario
    site['scenario'] = scenario

    ############### excel writer ###################
    spreadsheet_filename = 'Model_results_'+scenario+'.xlsx'
    path = Path.joinpath(cfg.OUTPUT_PATH, spreadsheet_filename)

    if os.path.isfile(path):
        print("Output file already exists.")
    
    else:
        writer = ExcelWriter(path)
        
        variables.to_excel(writer, 'variables')
        results.to_excel(writer, 'results')
        site.to_excel(writer, 'capacity')

        writer.save()
        writer.close()
    
    return 



########################### printing KPIs #######################################
def solution_kpis(variables, results):
    try:
        Site = pd.read_pickle(cfg.SITE_PATH)
        pv = pd.read_pickle(cfg.PRODUCT_VARIANT_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))
    
    site = pd.DataFrame(Site).T.reset_index()
    site.columns=['site','Period','cap','co','mb', 'u']
    site = site.drop(['site','co','mb', 'u'],axis=1)
    site['Period'] = site['Period'].astype(str)

    demand = pd.DataFrame(pv).T.reset_index()
    demand.columns=['Site', 'Customer', 'type', 'Product_id', 'Period', 'Demand']
    demand['Period'] = demand['Period'].astype(str)
    #demand = demand.groupby(['Customer']).sum()['Demand'].reset_index()

    ######################################################
    results = results[(results['Paretofront']!=1) & (results['iteration'] !=1)]
    index = [min(results['iteration']),  max(results['iteration']),  results[results['optimal']==1]['iteration'].values[0]]

    print('-------- KPIs for scenario comparison -----------------')
    print('Service level:  ')
    print('         first:       ', round(results[results['iteration'] == index[0]]['SL_total'].values[0],2))
    print('         last:        ', round(results[results['iteration'] == index[1]]['SL_total'].values[0],2))
    print('         Optimal:     ', round(results[results['iteration'] == index[2]]['SL_total'].values[0],2))


    print('Revenue:  ')
    print('         first:       ', round(results[results['iteration'] == index[0]]['Revenue'].values[0]))
    print('         last:        ', round(results[results['iteration'] == index[1]]['Revenue'].values[0]))
    print('         Optimal:     ', round(results[results['iteration'] == index[2]]['Revenue'].values[0]))


    ######################################################
    var = variables[variables['Period'] <'50']
    var = var.groupby(['iteration','Period']).sum()[['x','d']].reset_index()
    cap = var.merge(site, how = 'left', on='Period')
    cap['utilisation'] = cap['x']/cap['cap']
    var_first = cap[cap['iteration'] == index[0]]
    var_last = cap[cap['iteration'] == index[1]]
    var_optimal = cap[cap['iteration'] == index[2]]

    print('Average utilisation:  ')
    print('         first:       ', round(np.mean(var_first['utilisation']),5))
    print('         last:        ', round(np.mean(var_last['utilisation']),5))
    print('         Optimal:     ', round(np.mean(var_optimal['utilisation']),5))

    var2 = variables.loc[variables['iteration'].isin(index)]
    var2 = var2.merge(demand, how='left', on=['Site','Customer', 'Product_id', 'Period'])
    var2 = var2.groupby(['Site','Customer','Product_id','iteration']).sum()[['x','d','Demand']].reset_index()
    var2['fill'] = var2['x']/var2['Demand']

    print('Average utilisation:  ')
    print('         first:       ', 'mean: ',round(np.mean(var2[var2['iteration']==index[0]]['fill']),3), '  std: ', round(np.std(var2[var2['iteration']==index[0]]['fill']),3))
    print('         last:        ', 'mean: ',round(np.mean(var2[var2['iteration']==index[1]]['fill']),3), '  std: ', round(np.std(var2[var2['iteration']==index[1]]['fill']),3))
    print('         Optimal:     ', 'mean: ',round(np.mean(var2[var2['iteration']==index[2]]['fill']),3), '  std: ', round(np.std(var2[var2['iteration']==index[2]]['fill']),3))