# -*- coding: utf-8 -*-
from pathlib import Path
import os

# INPUT FILE
input_file = "Input_data_file.xlsx"

# Path to the main folder
ROOT_DIR = Path(__file__).absolute().parent.parent

# Data folder
DATA_PATH = Path.joinpath(ROOT_DIR, "Data")

# Directories for the raw data as dicts
INPUT_PATH = Path.joinpath(DATA_PATH, "Input")

# Output directory
OUTPUT_PATH = Path.joinpath(DATA_PATH, "Output")

# Output directory
DRAFT_PATH = Path.joinpath(DATA_PATH, "Draft")

# Graph path
GRAPH_PATH = Path.joinpath(DATA_PATH, "Graph")


#########   GLOBAL NAMES   ##########
filename_no_ext = ""
filename_pickle = ""

product_names = ""
line_names = ""
customer_names = ""
customer_names_all = ""
lines_site_name = ""
location_name = ""
time_intervals = ""
affiliate_type = ""

edges_name = ""

model_name = ""
model_vars = ""
model_name_vars = ""

# Parameters
customer_product = ""
customer_product_time = ""
site = ""
site_cust_prod = ""
site_product_time = ""
site_customer = ""
site_customer_time = ""
product_variant = ""
cust_prod_type = ""


##########   GLOBAL PATHS BASED ON FILENAME   ###########
FILENAME_PATH = ""
FILENAME_FOLDER_PATH = ""

DATA_PICKLE_PATH = ""

EDGES_PATH = ""

MODEL_PATH_VARS = ""
MODEL_PATH = ""

LINE_PATH = ""
PRODUCT_PATH = ""
CUSTOMER_PATH = ""
CUSTOMER_ALL_PATH = ""
LINES_SITE_PATH = ""
LOCATION_PATH = ""
TIME_PATH = ""
A_TYPE_PATH = ""


# Parameters
CUST_PROD_PATH = ""
CUST_PROD_TIME_PATH = ""
SITE_PATH = ""
SITE_CUST_PROD_PATH = ""
SITE_PROD_TIME_PATH = ""
SITE_CUST_PATH = ""
PRODUCT_VARIANT_PATH = ""
CUSTOMER_PRODUCT_TYPE_PATH = ""