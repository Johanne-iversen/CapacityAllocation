from BackBone import config as cfg
from pathlib import Path, PosixPath
import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import itertools
import numpy as np
import os


################ Build model #####################
def model_build(model_type, lp_filename='model.lp', start_week=None, included_weeks = None, delta1 = None, delta2 = None, iteration=None):
    """ 
    This function build the optimization model

    Filename dentoes the type of model used in the scenario
    """

    if start_week == None:
        start_week = 42
    if included_weeks == None:
        included_weeks = 4 
    iteration = iteration.__str__()

    ################ Model name definition ##################
    cfg.model_name = cfg.filename_no_ext + "_" + model_type + iteration + "_" + lp_filename


    ############### importing pahts, for input and output #################
    try:
        print("Using pickle files from: " + str(cfg.FILENAME_FOLDER_PATH))
        # indicien 
        J = pd.read_pickle(cfg.CUSTOMER_PATH)
        JJ = pd.read_pickle(cfg.CUSTOMER_ALL_PATH)
        I = pd.read_pickle(cfg.LOCATION_PATH)
        K = pd.read_pickle(cfg.PRODUCT_PATH)
        T = pd.read_pickle(cfg.TIME_PATH)
        P = pd.read_pickle(cfg.A_TYPE_PATH)

        # Variables 
        
        #edges_dict = pd.read_pickle(cfg.EDGES_PATH)
        CP_dict = pd.read_pickle(cfg.CUST_PROD_PATH)
        CPT_dict = pd.read_pickle(cfg.CUST_PROD_TIME_PATH)
        site_dict = pd.read_pickle(cfg.SITE_PATH)
        SC_dict = pd.read_pickle(cfg.SITE_CUST_PATH)
        PV_dict = pd.read_pickle(cfg.PRODUCT_VARIANT_PATH)
        SCP_dict = pd.read_pickle(cfg.SITE_CUST_PROD_PATH)
        CPP_dict = pd.read_pickle(cfg.CUSTOMER_PRODUCT_TYPE_PATH)

        # all data
        data = pd.read_pickle(cfg.DATA_PICKLE_PATH)

    except:
        print("One or more required files need to be placed in: " + str(cfg.FILENAME_FOLDER_PATH))
        raise


    if model_type == "1Iteration":

        cfg.MODEL_PATH = Path.joinpath(cfg.DRAFT_PATH, cfg.model_name)

        if os.path.isfile(cfg.MODEL_PATH):
            print("___________________________________________________________________________________________")
            print("______________FILE WITH THIS NAME ALREADY EXIST______________")
            print("___________________________________________________________________________________________")

        else:
            print("___________________________________________________________________________________________")
            print("______________BUILDING MODEL______________")
            print("___________________________________________________________________________________________")

        # Add variables
        cp_comb, Inv,  Bias, SS = gp.multidict(CP_dict)
        cpt_comb, gsf = gp.multidict(CPT_dict)
        st_dict, cap, CO, minB, mu = gp.multidict(site_dict)
        sc_dict, LT, week = gp.multidict(SC_dict)
        pv_comb, D = gp.multidict(PV_dict)
        cpp_comb, Px = gp.multidict(CPP_dict)

        # Build model
        m = gp.Model('RAP')

        ############### Decision varaibles ###############
        x = m.addVars(pv_comb, lb = 0, vtype = GRB.INTEGER,  name = 'x')
        v = m.addVars(cpt_comb, lb = 0, vtype = GRB.INTEGER, name = 'v')
        q = m.addVars(cpt_comb, lb = 0, vtype = GRB.INTEGER, name = 'q')
        qq = m.addVars(cpt_comb, vtype = GRB.BINARY, name = 'qq')
        y = m.addVars(pv_comb, vtype = GRB.BINARY, name = 'y')
        z = m.addVars(cpt_comb, vtype = GRB.BINARY, name = 'z')
        s = m.addVars(cpt_comb, lb = 0, ub = 1,  vtype = GRB.CONTINUOUS, name = "s")
        d = m.addVars(SCP_dict, lb = 0, vtype = GRB.INTEGER, name = 'd')
        sl = m.addVars(cpt_comb, vtype = GRB.CONTINUOUS, name = "sl")

        ############## Objective ###############
        m.setObjective(s.sum() - sl.sum(), GRB.MAXIMIZE)

        ############### Add constraints ###############

        # Balance constraints
        ## for the first period        
        balanse_0_con = m.addConstrs( \
            (Inv[j,k]  - gsf[j,k,t] - v[j,k,t] + q[j,k,t] == 0 \
                for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] == start_week]), \
                    name = "balance_0_con")
                        
        # ## for remaining periods
        balance_conL= m.addConstrs( \
            (v[j,k,t-1] - q[j,k,t-1] + gp.quicksum(x[i,j,p, k,t-week[i,j]] for i in I if t - week[i,j] >= start_week for p in P if p == 1)  - gsf[j,k,t] - v[j,k,t] + q[j,k,t] == 0\
                for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] > start_week]), \
                        name = "balance_conL")

        # binding q and v together 
        v_q_con = m.addConstrs( \
            (1000000*qq[j,k,t] >= v[j,k,t] for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'q_v_constraint')

        v_q_con2 = m.addConstrs(((qq[j,k,t] == 1) >> (q[j,k,t] == 0 ) for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), name ='v_q')

        # Supply and demand match stock holding affiliates
        demand_con = m.addConstrs(\
            (gp.quicksum(x[i,j,p,k,t] for t in T for p in P if p == 1) >= gp.quicksum(D[i,j,p,k,t] for t in T for p in P if p == 1) - d[i,j,k] \
                for i, j, k in [(key[0], key[1], key[2]) for key in SCP_dict]), \
                    name = 'demand_con')
        
        
        # Supply and demand match non-stockholding affiliates
        demand_con2 = m.addConstrs(\
            (x[i,j,p,k,t] == D[i,j,p,k,t] \
                for i, j, p, k, t in [(key[0], key[1], key[2], key[3], key[4]) for key in PV_dict if key[2] == 0]), \
                    name = 'demand_con2')

        # # Capacity constraint
        capacity_con = m.addConstrs( \
            (gp.quicksum(x[i,j,p,k,t] for j, p, k in [(key[0], key[1], key[2]) for key in CPP_dict]) <= cap[i,t]- \
                gp.quicksum(y[i,j,p,k,t] for j, p, k in [(key[0], key[1], key[2]) for key in CPP_dict]) * CO[i,t] * mu[i,t] for i,t in [(key[0], key[1]) for key in st_dict]), \
                name = 'capacity_con')

        # Minimum batch size 
        batchSize_con = m.addConstrs( \
            (x[i,j,p,k,t] >= minB[i,t] * y[i,j,p,k,t] for i, j, p, k, t in [(key[0], key[1], key[2], key[3],key[4]) for key in PV_dict if key[2] == 1] ), \
                name = 'batchSize_con')
        
        # Big M - reltaing x and y 
        big_M_con = m.addConstrs( \
            (x[i,j, p,k,t] <= 1000000 * y[i,j, p,k,t] for i, j, p, k, t in [(key[0], key[1], key[2], key[3], key[4]) for key in PV_dict]), \
                name = 'big_M_con')     

        # SS ? v
        safetyStock1 = m.addConstrs( \
            (v[j,k,t] - 1000000*(1-z[j,k,t]) <= SS[j,k] for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'SafetyStock_1')
        
        safetyStock2 = m.addConstrs( \
            (v[j,k,t] + 1000000*z[j,k,t] >= SS[j,k] for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'SafetyStock_2')

        # Service level
        serviclevel_con_z1 = m.addConstrs( \
            ((z[j,k,t] == 1) >> (SS[j,k] - v[j,k,t]  == SS[j,k] * (1 - s[j,k,t])) for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'servicelevel_con_z1') 
        
        serviclevel_con_z2 = m.addConstrs( \
            ((z[j,k,t] == 0) >> (s[j,k,t] == 1) for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] < (start_week + included_weeks - 3)]), \
                name = 'servicelevel_con_z2') 

        # Service level
        serviclevel_con = m.addConstrs( \
            (gsf[j,k,t] - q[j,k,t]  == gsf[j,k,t] * (1- sl[j,k,t]) for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] < (start_week + included_weeks - 3)]), \
                name = 'servicelevel_con') 

        ############## Save model ##############
        m.write(str(cfg.MODEL_PATH))


    ############### Epsilon constraint optimisation ########################
    elif model_type == "EpsilonConstraint":

        cfg.MODEL_PATH = Path.joinpath(cfg.DRAFT_PATH, cfg.model_name)

        if os.path.isfile(cfg.MODEL_PATH):
            print("___________________________________________________________________________________________")
            print("______________FILE WITH THIS NAME ALREADY EXIST______________")
            print("___________________________________________________________________________________________")

        else:
            print("___________________________________________________________________________________________")
            print("______________BUILDING MODEL______________")
            print("___________________________________________________________________________________________")

        # Add variables
        cp_comb, Inv,  Bias, SS = gp.multidict(CP_dict)
        cpt_comb, gsf = gp.multidict(CPT_dict)
        st_dict, cap, CO, minB, mu = gp.multidict(site_dict)
        sc_dict, LT, week = gp.multidict(SC_dict)
        pv_comb, D = gp.multidict(PV_dict)
        cpp_comb, Px = gp.multidict(CPP_dict)

        # Build model
        m = gp.Model('RAP')

        ############### Decision varaibles ###############
        x = m.addVars(pv_comb, lb = 0, vtype = GRB.INTEGER,  name = 'x')
        v = m.addVars(cpt_comb, lb = 0, vtype = GRB.INTEGER, name = 'v')
        q = m.addVars(cpt_comb, lb = 0, vtype = GRB.INTEGER, name = 'q')
        qq = m.addVars(cpt_comb, vtype = GRB.BINARY, name = 'qq')
        y = m.addVars(pv_comb, vtype = GRB.BINARY, name = 'y')
        z = m.addVars(cpt_comb, vtype = GRB.BINARY, name = 'z')
        s = m.addVars(cpt_comb, lb = 0, ub = 1,  vtype = GRB.CONTINUOUS, name = "s")
        d = m.addVars(SCP_dict, lb = 0, vtype = GRB.INTEGER, name = 'd')
        sl = m.addVars(cpt_comb, vtype = GRB.CONTINUOUS, name = "sl")

        ############## Objective ###############
        m.setObjective(gp.quicksum(gp.quicksum(x[i,j,p,k,t] for i,t in [(key[0], key[1]) for key in site_dict])* Px[j,p,k] for j,p,k in [(key[0], key[1], key[2]) for key in CPP_dict] ) - 1.8*d.sum(), GRB.MAXIMIZE)

        ############### Add constraints ###############

        # Balance constraints
        ## for the first period        
        balanse_0_con = m.addConstrs( \
            (Inv[j,k]  - gsf[j,k,t] - v[j,k,t] + q[j,k,t] == 0 \
                for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] == start_week]), \
                    name = "balance_0_con")
                        
        # ## for remaining periods
        balance_conL= m.addConstrs( \
            (v[j,k,t-1] - q[j,k,t-1] + gp.quicksum(x[i,j,p, k,t-week[i,j]] for i in I if t - week[i,j] >= start_week for p in P if p == 1)  - gsf[j,k,t] - v[j,k,t] + q[j,k,t] == 0\
                for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] > start_week]), \
                        name = "balance_conL")
        
        # binding q and v together 
        v_q_con = m.addConstrs( \
            (10000000*qq[j,k,t] >= v[j,k,t] for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'q_v_constraint')

        v_q_con2 = m.addConstrs(((qq[j,k,t] == 1) >> (q[j,k,t] == 0 ) for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), name ='v_q')


        # Supply and demand match stock holding affiliates
        demand_con = m.addConstrs(\
            (gp.quicksum(x[i,j,p,k,t] for t in T for p in P if p == 1) >= gp.quicksum(D[i,j,p,k,t] for t in T for p in P if p == 1) - d[i,j,k]  \
                for i, j, k in [(key[0], key[1], key[2]) for key in SCP_dict]), \
                    name = 'demand_con')
        
        # Supply and demand match non-stockholding affiliates
        demand_con2 = m.addConstrs(\
            (x[i,j,p,k,t] == D[i,j,p,k,t] \
                for i, j, p, k, t in [(key[0], key[1], key[2], key[3], key[4]) for key in PV_dict if key[2] == 0]), \
                    name = 'demand_con2')

        # # Capacity constraint
        capacity_con = m.addConstrs( \
            (gp.quicksum(x[i,j,p,k,t] for j, p, k in [(key[0], key[1], key[2]) for key in CPP_dict]) <= cap[i,t] - \
                gp.quicksum(y[i,j,p,k,t] for j, p, k in [(key[0], key[1], key[2]) for key in CPP_dict]) * CO[i,t] * mu[i,t] for i,t in [(key[0], key[1]) for key in st_dict]), \
                name = 'capacity_con') #cap[i,t]

        # Minimum batch size 
        batchSize_con = m.addConstrs( \
            (x[i,j,p,k,t] >= minB[i,t] * y[i,j,p,k,t] for i, j, p, k, t in [(key[0], key[1], key[2], key[3],key[4]) for key in PV_dict if key[2] == 1] ), \
                name = 'batchSize_con')
        
        # Big M - reltaing x and y 
        big_M_con = m.addConstrs( \
            (x[i,j, p,k,t] <= 10000000 * y[i,j, p,k,t] for i, j, p, k, t in [(key[0], key[1], key[2], key[3], key[4]) for key in PV_dict]), \
                name = 'big_M_con')     

        # SS ? v
        safetyStock1 = m.addConstrs( \
            (v[j,k,t] - 10000000*(1-z[j,k,t]) <= SS[j,k] for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'SafetyStock_1')
        
        safetyStock2 = m.addConstrs( \
            (v[j,k,t] + 10000000*z[j,k,t] >= SS[j,k] for j,k,t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'SafetyStock_2')

        # Service level
        serviclevel_con_z1 = m.addConstrs( \
            ((z[j,k,t] == 1) >> (SS[j,k] - v[j,k,t]  == SS[j,k] * (1 - s[j,k,t])) for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict]), \
                name = 'servicelevel_con_z1') 
        
        serviclevel_con_z2 = m.addConstrs( \
            ((z[j,k,t] == 0) >> (s[j,k,t] == 1) for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] < (start_week + included_weeks - 3)]), \
                name = 'servicelevel_con_z2') 

        # Service level
        serviclevel_con = m.addConstrs( \
            (gsf[j,k,t] - q[j,k,t]  == gsf[j,k,t] * (1- sl[j,k,t]) for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict if key[2] < (start_week + included_weeks - 3)]), \
                name = 'servicelevel_con') 

        # Supply limit - locked to epsilon
        epsilon_con = m.addConstr( \
            (sum(s[j,k,t] - sl[j,k,t] for j, k, t in [(key[0], key[1], key[2]) for key in CPT_dict]) >= delta1 - delta2), \
                name = 'epsilon_con') 

        ############## Save model ##############
        m.write(str(cfg.MODEL_PATH))

    return 



############ Read and solve LP model #################
def read_model(model_path, model_type1, iteration):
    '''
    Input a path to a .lp file.
    The directory starts in the Data folder.
    '''

    iteration = iteration.__str__()
    model_type = model_type1+iteration 
    model_name =  'Input_data_file_' + model_type + '_model' 
    cfg.model_name_vars = model_name + "_vars.pickle"
    cfg.MODEL_PATH = Path.joinpath(cfg.DATA_PATH, model_path)

    if model_type == '1Iteration'+iteration or model_type == 'EpsilonConstraint'+iteration:
        cfg.MODEL_PATH_VARS = Path.joinpath(cfg.DRAFT_PATH, cfg.model_name_vars)
    
    try:
        print("\n")
        print("\n")
        print("_________________________________________________")
        print("_________________READING LP FILE__________________")
        print("_____________________{}_____________________\n".format(model_type))
        m = gp.read(str(cfg.MODEL_PATH))
        print("________________________________________________")
        print("________________________________________________")
    except:
        print("________________________________________________\n")
        print("No .lp file found in: " + str(cfg.MODEL_PATH))
        print("________________________________________________\n")

    if os.path.isfile(cfg.MODEL_PATH_VARS):
        print("________________________________________________________________")
        print("_________________READING SOLUTION FROM PICKLE__________________")
        print("________________________________________________________________\n")
        variables = pd.read_pickle(cfg.MODEL_PATH_VARS)
    else:
        print("________________________________________________________________")
        print("_________________GENERATING SOLUTION PICKLE_____________________")
        print("________________________________________________________________\n")

        # Solve model
        m.optimize()

        # Get Variables
        variables = {}
        for v in m.getVars():
            index_list = tuple(v.varName.replace("[",",").replace(']','').split(","))

            variables[index_list] = v.x
        m.dispose()
        pd.to_pickle(variables, cfg.MODEL_PATH_VARS)

    return variables
