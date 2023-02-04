from BackBone import config as cfg
from pathlib import Path, PosixPath
import pandas as pd
import os
import itertools
import numpy as np
import datetime
import math
import glob


############## Configurate data path for all model input dicts #####################
def configure_paths(filename):


    cfg.FILENAME_PATH = Path.joinpath(cfg.INPUT_PATH, filename)
    cfg.filename_no_ext = filename.split(".")[0]
    cfg.filename_pickle = cfg.filename_no_ext + ".pickle"

    cfg.FILENAME_FOLDER_PATH = Path.joinpath(cfg.INPUT_PATH, cfg.filename_no_ext)
    cfg.DATA_PICKLE_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.filename_pickle)

    cfg.lines_site_name = str(cfg.filename_no_ext) + "_" + "lines_site_dic.pickle"
    cfg.line_name =  str(cfg.filename_no_ext) + "_" + "lines.pickle"
    cfg.location_name =  str(cfg.filename_no_ext) + "_" + "location.pickle"
    cfg.product_name =  str(cfg.filename_no_ext) + "_" + "product_lines.pickle"
    cfg.cus_name =  str(cfg.filename_no_ext) + "_" + "customer_list.pickle"
    cfg.cus_name_all =  str(cfg.filename_no_ext) + "_" + "customer_list_all.pickle"
    cfg.time_intervals = str(cfg.filename_no_ext) + "_" + "time_intervals.pickle"
    cfg.affiliate_type = str(cfg.filename_no_ext) + "_" + "affiliate_type.pickle"

    cfg.LINES_SITE_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.lines_site_name)
    cfg.LINE_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.line_name)
    cfg.LOCATION_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.location_name)
    cfg.PRODUCT_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.product_name)
    cfg.CUSTOMER_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.cus_name)
    cfg.CUSTOMER_ALL_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.cus_name_all)
    cfg.TIME_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.time_intervals)
    cfg.A_TYPE_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.affiliate_type)
    
    ## Parameters
    cfg.customer_product = cfg.filename_no_ext + "_" +  "customer_product.pickle"
    cfg.customer_product_time = cfg.filename_no_ext + "_" +  "customer_product_time.pickle"
    cfg.site = cfg.filename_no_ext + "_" +  "site.pickle"
    cfg.site_cust_prod = cfg.filename_no_ext + "_" +  "site_cust_prod.pickle"
    cfg.site_product_time = cfg.filename_no_ext + "_" + "site_product_time.pickle"
    cfg.site_customer = cfg.filename_no_ext + "_" + "site_customer.pickle"
    cfg.product_variant = cfg.filename_no_ext + "_" + "product_variant.pickle"
    cfg.cust_prod_type = cfg.filename_no_ext + "_" + "cust_prod_type.pickle"

    cfg.CUST_PROD_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.customer_product)
    cfg.CUST_PROD_TIME_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.customer_product_time)
    cfg.SITE_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.site)
    cfg.SITE_CUST_PROD_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.site_cust_prod)
    cfg.SITE_PROD_TIME_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.site_product_time)
    cfg.SITE_CUST_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.site_customer)
    cfg.PRODUCT_VARIANT_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.product_variant)
    cfg.CUSTOMER_PRODUCT_TYPE_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.cust_prod_type)

    ## EDGES
    cfg.edges_name = cfg.filename_no_ext + "_edges.pickle"
    cfg.EDGES_PATH = Path.joinpath(cfg.FILENAME_FOLDER_PATH, cfg.edges_name)

    return

############## Load input data into dictonaries ##################
def load_input(calc_edges = True, start_week = None, included_weeks = None):
    """
    This function takes a spreadsheet as input and returns the data.
    If there exists a pickle file with the same name as the spreadsheet this file
    is loaded instead of the spreadsheet to reduce computation time.
    """

    if os.path.isfile(cfg.DATA_PICKLE_PATH):
        print("Loading input data from pickle.")
        data = pd.read_pickle(cfg.DATA_PICKLE_PATH)
    else:
        print("Reading input data from excel file.")
        data = pd.read_excel(cfg.FILENAME_PATH, sheet_name= None)

        if not os.path.exists(cfg.FILENAME_FOLDER_PATH):
            os.makedirs(cfg.FILENAME_FOLDER_PATH)
        pd.to_pickle(data, cfg.DATA_PICKLE_PATH)

    if start_week == None:
        start_week = datetime.date.today().isocalendar()[1]+1
    if included_weeks == None:
        included_weeks = 4


    ################################# PARAMETERS ####################################################
    ## -----------------------------------------------------------------------------------------------------------
    # keys: (Customer, Product)                 values: (inventory, Forecast bias)

    if os.path.isfile(cfg.CUST_PROD_PATH):
        print("Loading cust_prod from pickle: " + str(cfg.CUST_PROD_PATH))
        cust_prod_dict = pd.read_pickle(cfg.CUST_PROD_PATH)
    elif calc_edges:
        print("Calculating customer product edges from input data: " + str(cfg.DATA_PICKLE_PATH))
        cust_prod = []
        cust_prod_df = []

        for index, row in data['PV_SH'].iterrows():
            
            # Keys
            customer = row['Customer_Id']
            product = row['Product_id']

            # Values
            inventory = data['Affiliate_inventory'][(data['Affiliate_inventory']['Country'] == customer) & (data['Affiliate_inventory']['Product_id'] == product)]['Sum_QuantityU3'].sum()
            SafetyStock = data['SafetyStock'][(data['SafetyStock']['Customer_Id'] == customer) & (data['SafetyStock']['Product_id'] == product)]['Avg_SafetyStockQty'].values[0]
            beta = np.random.normal(0, 0.05)

            cust_prod = [str(customer), product, [inventory, beta, SafetyStock]]
            cust_prod_df.append(cust_prod)
            
        cust_prod_dict = {tuple(CP[:2]): CP[2] for CP in cust_prod_df}
        pd.to_pickle(cust_prod_dict, cfg.CUST_PROD_PATH)

    ##---------------------------------------------------------------------------------------------------------------------
    # keys: (Customer, Product, Time)             values: (GSF, )

    if os.path.isfile(cfg.CUST_PROD_TIME_PATH):
        print("Loading cust_prod_time from pickle: " + str(cfg.CUST_PROD_TIME_PATH))
        cust_prod_time_dict = pd.read_pickle(cfg.CUST_PROD_TIME_PATH)
    elif calc_edges:
        print("Calculating customer product edges from input data: " + str(cfg.DATA_PICKLE_PATH))
        cpt = []
        cust_prod_time_df = []

        for index, row in data['PV_SH'].iterrows():

            # Keys
            customer = row['Customer_Id']
            product = row['Product_id']

            #leadtime = data['LeadTimes'][(data['LeadTimes']['Customer_Id'] == customer)]['Avg_leadtime'].values[0]
            #dist_weeks = math.ceil(leadtime / 7)

            for i in range(included_weeks):
                row['time'] = start_week + i #+ dist_weeks
                period = row['time']
                gsf = data['GSF'][(data['GSF']['Country_ID'] == customer) & (data['GSF']['Product_id'] == product) & (data['GSF']['NOVO_WEEK'] == period)]['Sum_Volume_U3'].sum()

                cpt = [str(customer), product, period, [gsf]]
                cust_prod_time_df.append(cpt)
            
        cust_prod_time_dict = {tuple(CP[:3]): CP[3] for CP in cust_prod_time_df}
        pd.to_pickle(cust_prod_time_dict, cfg.CUST_PROD_TIME_PATH)

    ## ---------------------------------------------------------------------------------------------------------------------
    # keys: (Site, time)                         values: (Capapacity, utilization, Changover, batchsize_min, batchsize_ max)

    if os.path.isfile(cfg.SITE_PATH):
        print("Loading site from pickle: " + str(cfg.SITE_PATH))
        site_dict = pd.read_pickle(cfg.SITE_PATH)
    elif calc_edges:
        print("Calculating customer product edges from input data: " + str(cfg.DATA_PICKLE_PATH))
        site = []
        site_df = []

        for index, row in data['Line_Capacity'].iterrows():

            # Keys
            location = row['Location']
        
            # Values
            for i in range(included_weeks):
                row['time'] = start_week + i
                period = row['time']
                capacity = data['Line_Capacity'][(data['Line_Capacity']['Location'] == location) & (data['Line_Capacity']['NOVO_WEEK'] == period)]['Capacity'].sum()
                change_over = data['Line_Capacity'][(data['Line_Capacity']['Location'] == location) & (data['Line_Capacity']['NOVO_WEEK'] == period)]['Change_over'].sum()
                MinBSize = data['Line_Capacity'][(data['Line_Capacity']['Location'] == location) & (data['Line_Capacity']['NOVO_WEEK'] == period)]['Min_batch'].sum()
                utilization = data['Line_Capacity'][(data['Line_Capacity']['Location'] == location) & (data['Line_Capacity']['NOVO_WEEK'] == period)]['Line_speed'].sum()

                site = [str(location), period, [capacity, change_over, MinBSize, utilization]]
                site_df.append(site)
            
        site_dict = {tuple(S[:2]): S[2] for S in site_df}
        pd.to_pickle(site_dict, cfg.SITE_PATH)

    ## ----------------------------------------------------------------------------------------------------------------------
    # keys: (Site, customer, product)                Values: N/A
    if os.path.isfile(cfg.SITE_CUST_PROD_PATH):
        print("Loading site customer product from pickle: " + str(cfg.SITE_CUST_PROD_PATH))
        site_cust_prod_dict = pd.read_pickle(cfg.SITE_CUST_PROD_PATH)
    elif calc_edges:
        print("Calculating site customer product edges from input data: " + str(cfg.DATA_PICKLE_PATH))
        scp = []
        scp_df = []

        for index, row in data['PV_SH'].iterrows():

            # Keys
            location = row['Location']
            customer = row['Customer_Id']
            product = row['Product_id']


            scp = [str(location), str(customer), product]
            scp_df.append(scp)
            
        site_cust_prod_dict = {tuple(S[:3]) for S in scp_df}
        pd.to_pickle(site_cust_prod_dict, cfg.SITE_CUST_PROD_PATH)

        
    ## ----------------------------------------------------------------------------------------------------------------------
    # keys: (Site, customer)                 Values: (lead time)
    if os.path.isfile(cfg.SITE_CUST_PATH):
        print("Loading site to customer from pickle: " + str(cfg.SITE_CUST_PATH))
        site_customer_dict = pd.read_pickle(cfg.SITE_CUST_PATH)
    elif calc_edges:
        print("Calculating distribution times edges from input data: " + str(cfg.DATA_PICKLE_PATH))
        sc = []
        sc_df = []

        for index, row in data['LeadTimes'].iterrows():

            # Keys

            location = row['Location']
            customer = row['Customer_Id']

            # Value
            leadtime = data['LeadTimes'][(data['LeadTimes']['Customer_Id'] == customer) & (data['LeadTimes']['Location'] == location)]['Avg_leadtime'].values[0]
            dist_weeks = math.ceil(leadtime / 7)

            sc = [str(location), str(customer), [leadtime, dist_weeks]]
            sc_df.append(sc)

        site_customer_dict = {tuple(CP[:2]): CP[2] for CP in sc_df}
        pd.to_pickle(site_customer_dict, cfg.SITE_CUST_PATH)

    ## ----------------------------------------------------------------------------------------------------------------------
    # keys: (Site, Customer, Product, Time)                 Values: (Order qty)
    if os.path.isfile(cfg.PRODUCT_VARIANT_PATH):
        print("Loading product_variant from pickle: " + str(cfg.PRODUCT_VARIANT_PATH))
        product_variant_dict = pd.read_pickle(cfg.PRODUCT_VARIANT_PATH)
    elif calc_edges:
        print("Calculating product variant edges from input data: " + str(cfg.DATA_PICKLE_PATH))
        pv = []
        pv_df = []

        for index, row in data['PV_unique'].iterrows():

            # Keys
            location = row['Location']
            customer = row['Customer_Id']
            product = row['Product_id']
            a_type = row['Affiliate_type']
            #a_type = row['Affiliate_type']

            for i in range(included_weeks):
                row['time'] = start_week+i
                period = row['time']
                order_qty = data['Upcoming_Orders'][(data['Upcoming_Orders']['Country_ID'] == customer) & \
                    (data['Upcoming_Orders']['Product_id'] == product) & \
                    (data['Upcoming_Orders']['NOVO_WEEK'] == period) & \
                    (data['Upcoming_Orders']['Production_Plant'] == location)]\
                    ['Order_Quantity_U3'].sum()

                pv = [str(location), str(customer), a_type, product, period, [order_qty]]
                pv_df.append(pv)
            
        product_variant_dict = {tuple(CP[:5]): CP[5] for CP in pv_df}
        pd.to_pickle(product_variant_dict, cfg.PRODUCT_VARIANT_PATH)

    ## ----------------------------------------------------------------------------------------------------------------------
    # keys: (Customer, Product, affiliate type)                 Values: - 
    if os.path.isfile(cfg.CUSTOMER_PRODUCT_TYPE_PATH):
        print("Loading product_variant from pickle: " + str(cfg.CUSTOMER_PRODUCT_TYPE_PATH))
        cust_prod_type_dict = pd.read_pickle(cfg.CUSTOMER_PRODUCT_TYPE_PATH)
    elif calc_edges:
        print("Calculating product variant edges from input data: " + str(cfg.DATA_PICKLE_PATH))
        cpa = []
        cpa_df = []

        for index, row in data['PV_unique'].iterrows():

            # Keys
            customer = row['Customer_Id']
            product = row['Product_id']
            a_type = row['Affiliate_type']

            pref = data['PrefCost'][(data['PrefCost']['Customer_Id'] == customer) & (data['PrefCost']['Product_id'] == product)]['Pref_cost'].values[0]

            cpa = [str(customer), a_type, product, [pref]]
            cpa_df.append(cpa)
            
        cust_prod_type_dict = {tuple(CP[:3]): CP[3] for CP in cpa_df}
        pd.to_pickle(cust_prod_type_dict, cfg.CUSTOMER_PRODUCT_TYPE_PATH)




    ################################# VARIABLES INDICES ###################################################
    # Lines at site 
    lines_site_dic = data['Line_Capacity'].groupby('Location')['Line_Id'].apply(list).to_dict()
    pd.to_pickle(lines_site_dic, cfg.LINES_SITE_PATH)

    # Pack Lines
    line_sites =  data['Line_Capacity'][(data['Line_Capacity']['Process'] == 'PACK')]['Line_Id'].to_list()
    line_site = list(set(line_sites))
    pd.to_pickle(line_site, cfg.LINE_PATH)

    # Locations
    location =  list(data['Line_Capacity']['Location'].unique())
    location = [str(x) for x in location]
    pd.to_pickle(location, cfg.LOCATION_PATH)

    # Unique customers
    customer_list = list(data['PV_unique'][data['PV_unique']['Affiliate_type'] == 1]['Customer_Id'].unique())
    customer_list = [str(x) for x in customer_list]
    pd.to_pickle(customer_list, cfg.CUSTOMER_PATH)

     # Unique customers
    customer_list_all = list(data['PV_unique']['Customer_Id'].unique())
    customer_list_all = [str(x) for x in customer_list]
    pd.to_pickle(customer_list_all, cfg.CUSTOMER_ALL_PATH)

    # Unique Products
    product_list = list(data['PV_unique']['Product_id'].unique())
    pd.to_pickle(product_list, cfg.PRODUCT_PATH)

    affiliate_type = list(data['PV_unique']['Affiliate_type'].unique())
    pd.to_pickle(affiliate_type, cfg.A_TYPE_PATH)

    time_intervals = []
    for i in range(included_weeks):
        time_intervals.append(start_week + i)
    pd.to_pickle(time_intervals,cfg.TIME_PATH)



def output_loader(name):
    output_path = os.path.join(cfg.OUTPUT_PATH)
    files = glob.glob(output_path+ "/Model_results_"+name+".xlsx")

    # create a new dataframe to store the
    var_list = []
    cap_list = []
    result_list = []
    for f in files:
            var_list.append(pd.read_excel(f, sheet_name='variables'))
            cap_list.append(pd.read_excel(f, sheet_name='capacity'))
            result_list.append(pd.read_excel(f, sheet_name='results'))
    # merged excel file.
    var_merged = pd.DataFrame()
    cap_merged = pd.DataFrame()
    result_merged = pd.DataFrame()

    # appends the data into the merged file
    for f in var_list:
        var_merged = var_merged.append(f)
    for f in cap_list:
        cap_merged = cap_merged.append(f)
    for f in result_list:
        result_merged = result_merged.append(f)

    return var_merged, cap_merged, result_merged
    

def output_loader_all():
    output_path = os.path.join(cfg.OUTPUT_PATH)
    files = glob.glob(output_path+ "/*.xlsx")

    var_list = []
    cap_list = []
    result_list = []
    for f in files:
            var_list.append(pd.read_excel(f, sheet_name='variables'))
            cap_list.append(pd.read_excel(f, sheet_name='capacity'))
            result_list.append(pd.read_excel(f, sheet_name='results'))

 
    # create a new dataframe to store the
    # merged excel file.
    var_merged = pd.DataFrame()
    cap_merged = pd.DataFrame()
    result_merged = pd.DataFrame()

    # appends the data into the merged file
    for f in var_list:
        var_merged = var_merged.append(f)
    for f in cap_list:
        cap_merged = cap_merged.append(f)
    for f in result_list:
        result_merged = result_merged.append(f)

    return var_merged, cap_merged, result_merged

