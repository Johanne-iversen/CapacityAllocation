from BackBone import config as cfg
from pathlib import Path, PosixPath
from BackBone.tools import Input_tools
import pandas as pd
import numpy as np

# Plot imports
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
sns.set_style("darkgrid")
from matplotlib import cm, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# radar packages
from matplotlib.patches import Circle, RegularPolygon
from matplotlib.path import Path
from matplotlib.projections.polar import PolarAxes
from matplotlib.projections import register_projection
from matplotlib.spines import Spine
from matplotlib.transforms import Affine2D


################ Plot: the paratero front #######################
def paretofront(results,fig,ax):   
    '''
    Plotting all solutions of the epsilon optimisation, collored by prunning
    '''
    color = ['salmon' if x == 1 else 'fuchsia' if x==2 else '#04D8B2' for x in results['Paretofront']]
    ax.scatter((results['SL_total']), results['Revenue'], c=color, s=35)
    ax.set_ylim(min(results['Revenue'][1:]-100000),max(results['Revenue'])+100000)
    ax.set_title('Pareto front')
    ax.set_xlabel('Service level')
    ax.set_ylabel('solution revenue')

    c1 = mpatches.Patch(color='salmon', label = 'Non-Optimal')
    c2 = mpatches.Patch(color='fuchsia', label = 'Optimal')
    c3 = mpatches.Patch(color='aquamarine', label = 'Pareto-Optimal')

    ax.legend(handles=[c1,c2,c3])
    fig.show()
    
    return


################ Plot:  Pareto solutions of all scenarios ########################
def pareto_compare(results): #input result_merged   
    '''
    Plotting all solutions of the epsilon optimisation, collored by prunning
    '''
    scenario = list(results['scenario'].unique())
    colors = ['#FFA630','#76E7CD','#F05D5E','#0F7173','#2F3061','#78C0E0','#A06CD5']
    col = {scenario[i]: colors[i] for i in range(len(scenario))}
    col2 = pd.DataFrame.from_dict(col, orient = 'index').reset_index()
    col2 = col2.rename(columns={0:'HEX', 'index':'scenario'})

    results = results[results['Paretofront'] != 1]
    results = results.merge(col2, how='left', on='scenario')
    results.loc[results['optimal']==1, ['HEX']] = 'magenta'


    plt.scatter((results['SL_total']), results['Revenue'], c=results['HEX'], s=35)#c=results['scenario'].map(col2), s=35)
    plt.ylim(min(results['Revenue'][1:]-1000000),max(results['Revenue'])+1000000)
    plt.title('Pareto front')
    plt.xlabel('Service level')
    plt.ylabel('solution revenue')
    
    # The following two lines generate custom fake lines that will be used as legend entries:
    markers = [plt.Line2D([0,0],[0,0],color=color, marker='o', markersize=10, linestyle='') for color in col.values()]
    plt.legend(markers, col.keys(), numpoints=1)
    plt.show()

    return 


################## Plot: weekly capacity, demand and capacity allocation  #################
def capacity_allocation_split(variables, cap, iteration, fig, ax):
    '''
    This function visualise the capacity, demand and capacity allocation for both stockholding 
    and non stockholding affiliates, for a specific iteration in a scenario
    '''
   
    cap['Period'] = cap['Period'].astype(str)
    variables['Period'] = variables['Period'].astype(str)
    variables['type'] = variables['type'].astype(str)
    variables = variables.drop(['Unnamed: 0'],axis=1)

    variables = variables.loc[variables['iteration']==iteration]
    var_df = variables.groupby(['Period','type']).sum()[['x']].reset_index()
    df = var_df.pivot_table('x','Period', 'type').reset_index()
    df_demand = variables.groupby('Period').sum()['Demand']

    df = df.merge(cap, how='left', on='Period')
    df = df.merge(df_demand, how='left', on='Period')
    df.columns = ['Period', 'Non', 'SH','index', 'Capacity','scenario', 'Demand']

    pal = sns.color_palette("Set1")
    ax.plot(df.Period, df.Capacity, c='#04D8B2', linestyle='-',linewidth=3, label='Capacity')
    ax.plot(df.Period, df.Demand,  c='#FF81C0', linestyle='-', linewidth=3, label='Demand' )
    ax.plot(df.Period, df.SH,  c='#069AF3', linestyle='-', linewidth=3, label='StockHolding')
    ax.plot(df.Period, df.Non, c='#C79FEF', linestyle='-', linewidth=3, label='Non-StockHolding')
    
    ax.set_title('Capacity allocation across planning periods')
    ax.set_xlabel('Planning weeks')
    ax.set_ylabel('Volume (pcs.)')
    ax.legend(loc='upper right')
    fig.show()
    
    return



################ Plot: the capacity utilisation ##################
def cap_utilisation(variables, fig, ax):
    '''
    This function visualise the capacity allocation for each planning week across all iteration
    '''
    try:
        Site = pd.read_pickle(cfg.SITE_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))
    
    site = pd.DataFrame(Site).T.reset_index()
    site.columns=['site','Period','cap','co','mb', 'u']
    site = site.drop(['site','co','mb', 'u'],axis=1)
    site['Period'] = site['Period'].astype(str)
    
    df = variables.loc[variables['iteration']!=1]
    df = df.groupby(['Period','iteration']).sum()['x'].reset_index()
    iter = list(df['iteration'].unique())
    util = df.merge(site, how='left', on=['Period'])
    util['utilisation']=util['x']/util['cap']

    my_cmap = cm.winter_r
    my_norm = colors.Normalize(vmin=min(iter), vmax=max(iter))

    for i in range(len(iter)):
        ax.plot(util[util['iteration'] == iter[i]]['Period'],util[util['iteration'] == iter[i]]['utilisation'], color=my_cmap(my_norm(iter[i])))
    ax.set_title('Capacity utilisation over all iterations')
    ax.set_xlabel('Planning period')
    ax.set_ylabel('Capacity utilisation')
    fig.colorbar(cm.ScalarMappable(norm=my_norm, cmap=my_cmap), ax=ax, orientation="vertical", label="iteration")
    fig.show()

    return

############## Plot: Capacity utilisation over solution revenue #####################
def util_profit(variables, results, fig, ax):
    '''
    This function plots the each iterations average capacity utilisation and revenue
    coloured according to  the solutions service level
    '''
    try:
        Site = pd.read_pickle(cfg.SITE_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))
    
    site = pd.DataFrame(Site).T.reset_index()
    site.columns=['site','Period','cap','co','mb', 'u']
    site = site.drop(['site','co','mb', 'u'],axis=1)
    site['Period'] = site['Period'].astype(str)

    df = variables.loc[variables['iteration']!=1]
    df = df.groupby(['Period','iteration']).sum()['x'].reset_index()
    util = df.merge(site, how='left', on=['Period'])
    util['utilisation']=util['x']/util['cap']
    util = pd.DataFrame(util.groupby('iteration').mean()['utilisation'].reset_index())
    df1 = pd.concat([util,results[1:]], axis=1)

    my_cmap = cm.cool
    my_norm = colors.Normalize(vmin=df1["S"].min(), vmax=df1["S"].max())

    ax.scatter(df1['Revenue'], df1['utilisation'],color=my_cmap(my_norm(df1["S"])))# color =plt.cm.cool(df1['S']) )
    ax.set_title(' Relation between capacity utilisation and solution revenue')
    ax.set_xlabel('Solution revenue')
    ax.set_ylabel('Average Capacity utilisation')
    fig.colorbar(cm.ScalarMappable(norm=my_norm, cmap=my_cmap), ax=ax, orientation="vertical", label="Service level")

    fig.show()
    
    return



############## Plot: fillrate across iterations for product variants ###############
def demand_fulfillment_dev1(variable_df):
    '''
    This function visualise the fill rate for selected product variants orders, across all iterations in a scenario
    '''
    try:
        pv = pd.read_pickle(cfg.PRODUCT_VARIANT_PATH)
        cpp = pd.read_pickle(cfg.CUSTOMER_PRODUCT_TYPE_PATH)
    except:
        print("Files needed missing in directory: " + str(cfg.FILENAME_FOLDER_PATH))

    demand = variable_df[variable_df['iteration']==1]
    demand = demand.groupby(['Customer','Product_id']).sum()['Demand'].reset_index()
    demand['Customer'] = demand['Customer'].astype(str)
    demand = demand[demand['Demand'] > 0]

    price = pd.DataFrame(cpp).T.reset_index()
    price.columns=['Customer', 'type', 'Product_id', 'price']
    price = price.groupby(['Customer','Product_id']).mean()['price'].reset_index()

    variable_df = variable_df.groupby(['Customer','Product_id','iteration']).agg({'x':'sum','s':'mean'}).reset_index().fillna(1)
    variable_df['Customer'] = variable_df['Customer'].astype(str)
    demand_df = demand.merge(variable_df, how='left', on=['Customer', 'Product_id'])
    demand_df['fill_rate'] = demand_df['x'] /demand_df['Demand']
    demand_df = demand_df[demand_df['iteration']>1]
    Customers = demand.groupby(['Customer','Product_id']).count().reset_index()
    Customers['Product_id_naming'] = Customers['Product_id'].str.split('_').to_list()
    customer_text = Input_tools.customers()
    max_fill_rate = max(demand_df['fill_rate'])
    print(max_fill_rate)

    my_cmap = cm.cool
    my_norm = colors.Normalize(vmin=0, vmax=demand_df["s"].max())
    fig, axes = plt.subplots(5,8, figsize=(21,14))
    index = 0

    axes.flat[-1].set_visible(False)
    axes.flat[-2].set_visible(False)
    axes.flat[0].set_ylabel('Fill rate')
    axes.flat[8].set_ylabel('Fill rate')
    axes.flat[16].set_ylabel('Fill rate')
    axes.flat[24].set_ylabel('Fill rate')
    axes.flat[32].set_ylabel('Fill rate')

    
    for ax in axes.flatten()[-10:-2]:
        ax.set_xlabel('Iterations')

    for ax in axes.flatten()[:-2]:
        dist_df = demand_df[(demand_df['Customer'] == Customers['Customer'][index]) & (demand_df['Product_id'] == Customers['Product_id'][index])]
        
        ax.scatter(dist_df['iteration'], dist_df['fill_rate'], s=8, color=my_cmap(my_norm(dist_df["s"])))
        ax.set_title(customer_text[customer_text['Customer_Id'] == Customers['Customer'][index]]['CountryText'].values[0] + '\n ' +  Customers['Product_id_naming'][index][2] + ' ' + Customers['Product_id_naming'][index][3]+ ' ' + Customers['Product_id_naming'][index][4])
        ax.set_ylim([0,max_fill_rate+0.1])
        ax.text(85,2.5, "\n Revenue {:0.2f}\n".format(price.loc[(price['Customer']==Customers['Customer'][index]) & (price['Product_id']==Customers['Product_id'][index])].iat[0,2]), verticalalignment = 'center', horizontalalignment = 'right', fontsize=11,
                    bbox=dict(boxstyle = 'square', facecolor='white', edgecolor='black', pad=0, alpha=0.4))
        index += 1 
       
    fig.subplots_adjust(right=0.8, hspace=0.7, wspace=0.4)
    cbar_ax = fig.add_axes([0.84,0.1,0.02,0.8])
    cbar = fig.colorbar(cm.ScalarMappable(norm=my_norm, cmap=my_cmap), cax=cbar_ax)
    cbar.set_label('Avarage service level ')
    plt.show()

    return 


########################### spiserwed plots #####################################
################ helpers ################

######## Standardise: total solution service level #############
def polar_SL(result, index, scenario):
    '''
    Standardise the total solution service level for a given iteration, in relation to the global min and max
    obtained across all acenarios
    '''
    min = result['SL_total'].min()
    max = result['SL_total'].max()
    t_max = 1
    t_min = 0

    res = result[result['scenario']==scenario]
    r = []
    for i in range(len(index)):
        r_sl = ((res[res['iteration']==index[i]]['SL_total'].values[0]-min)/(max-min))*(t_max-t_min)+t_min    # service level nstandardised between 0.2-1
        r.append(r_sl)
    return r


######## Standardise: solution revenue #############
def polar_Rev(result,index, scenario):
    '''
    Standardise the revenue for a given iteration, in relation to the global min and max
    obtained across all acenarios
    '''
    min = result['Revenue'].min()
    max = result['Revenue'].max()
    t_max = 1
    t_min = 0

    res = result[result['scenario']==scenario]
    r = []
    for i in range(len(index)):
        r_R = ((res[res['iteration']==index[i]]['Revenue'].values[0]-min)/(max-min))*(t_max-t_min)+t_min    # solution revenue standardised between 0.2-1
        r.append(r_R)
    return r


######## Standardise: Average solution fill rate #############
def polar_fillrate(var, scenario, index):
    '''
    Standardise the avarage fill rate for a given iteration, in relation to the global min and max
    obtained across all acenarios
    '''
    var= var.groupby(['scenario','Customer','Product_id','iteration']).sum()[['x','Demand']].reset_index()
    var = var[(var['Demand']>0)]
    var['fillrate'] = var['x']/var['Demand']
    var1 = var.groupby(['scenario','iteration']).mean()['fillrate'].reset_index()

    min1 = min(var1['fillrate'])
    max1 = max(var1['fillrate'])

    var1 = var1[var1['scenario']==scenario]
    r_fill = []
    for i in range(len(index)):
        check = var1[var1['iteration']==index[i]]['fillrate'].values[0]
        r_fill.append(((check-min1)/(max1-min1)*(1-0)+0))

    return  r_fill

    
######## Standardise: capacity utilisation #############
def polar_caputil(var, index, scenario):
    '''
    Standardise the capacity utilisation for a given iteration, in relation to the global min and max
    obtained across all acenarios
    '''
    var = var.groupby(['scenario','iteration']).sum()['y'].reset_index()
    var = var[var['iteration'] !=1]
    max1 = max(var['y'])
    min1 = min(var['y'])
    t_min = 0
    t_max = 1

    var = var[var['scenario']== scenario]
    r_cap = []
    for i in range(len(index)):
        r = 1-((var[var['iteration']== index[i]]['y'].values[0] - min1)/(max1-min1))*(t_max-t_min)+t_min
        r_cap.append(r)
    
    return  r_cap


############ Radar plot helper #################
def radar_factory(num_vars, frame='circle'):
    """
    Create a radar chart with `num_vars` axes.

    This function creates a RadarAxes projection and registers it.

    Parameters
    ----------
    num_vars : int
        Number of variables for radar chart.
    frame : {'circle', 'polygon'}
        Shape of frame surrounding axes.

    """
    # calculate evenly-spaced axis angles
    theta = np.linspace(0, 2*np.pi, num_vars, endpoint=False)

    class RadarTransform(PolarAxes.PolarTransform):

        def transform_path_non_affine(self, path):
            # Paths with non-unit interpolation steps correspond to gridlines,
            # in which case we force interpolation (to defeat PolarTransform's
            # autoconversion to circular arcs).
            if path._interpolation_steps > 1:
                path = path.interpolated(num_vars)
            return Path(self.transform(path.vertices), path.codes)

    class RadarAxes(PolarAxes):

        name = 'radar'
        PolarTransform = RadarTransform

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # rotate plot such that the first axis is at the top
            self.set_theta_zero_location('N')

        def fill(self, *args, closed=True, **kwargs):
            """Override fill so that line is closed by default"""
            return super().fill(closed=closed, *args, **kwargs)

        def plot(self, *args, **kwargs):
            """Override plot so that line is closed by default"""
            lines = super().plot(*args, **kwargs)
            for line in lines:
                self._close_line(line)

        def _close_line(self, line):
            x, y = line.get_data()
            # FIXME: markers at x[0], y[0] get doubled-up
            if x[0] != x[-1]:
                x = np.append(x, x[0])
                y = np.append(y, y[0])
                line.set_data(x, y)

        def set_varlabels(self, labels):
            self.set_thetagrids(np.degrees(theta), labels)

        def _gen_axes_patch(self):
            # The Axes patch must be centered at (0.5, 0.5) and of radius 0.5
            # in axes coordinates.
            if frame == 'circle':
                return Circle((0.5, 0.5), 0.5)
            elif frame == 'polygon':
                return RegularPolygon((0.5, 0.5), num_vars,
                                      radius=.5, edgecolor="k")
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

        def _gen_axes_spines(self):
            if frame == 'circle':
                return super()._gen_axes_spines()
            elif frame == 'polygon':
                spine = Spine(axes=self,
                              spine_type='circle',
                              path=Path.unit_regular_polygon(num_vars))
                spine.set_transform(Affine2D().scale(.5).translate(.5, .5)
                                    + self.transAxes)
                return {'polar': spine}
            else:
                raise ValueError("Unknown value for 'frame': %s" % frame)

    register_projection(RadarAxes)
    return theta

################# Radar plot data adjustments ##################
def polar_data(var, results, scenario):
    '''
    This function forms a dataframe containing the standardised values of all four performance measures
    '''
    pareto = results[results['scenario']==scenario]
    pareto = pareto[pareto['iteration']>1]
    list = pareto[pareto['Paretofront']!=1]['iteration'].to_list()
    first = min(list)
    last = max(list)
    optimal = pareto[pareto['Paretofront']==2]['iteration'].values[0]
    index = [first, last, optimal]

    data = [['Service Level', 'Revenue', 'Cost','Avg. fill rate'],
            (scenario, np.transpose([polar_SL(results, index, scenario), polar_Rev(results, index,scenario), polar_caputil(var, index, scenario), polar_fillrate(var, scenario, index)]))
            ]
    return data, index


########## Plot: radar ##############
def norm_resutls2(var, results, scenario, ax, fig):
    '''
    This function take the normalised preformance measered and visualise it in in a radar plot
    containing the best solution depending on objective 1, 2 and optimality
    '''
    N = 4
    theta = radar_factory(N, frame='polygon')
    data, index = polar_data(var,results, scenario)
    spoke_labels = data.pop(0)

    colors = ['g', 'r', 'b']

    # Plot the four cases from the example data on separate axes
    for (title, case_data) in  data:
        ax.set_rgrids([0.2, 0.4, 0.6, 0.8])
        ax.set_title(title, weight='bold', size='large', position=(0.5, 1.1),
                     horizontalalignment='center', verticalalignment='center')
        for d, color in zip(case_data, colors):
            ax.plot(theta, d, color=color)
            ax.fill(theta, d, facecolor=color, alpha=0.25, label='_nolegend_')
        ax.set_varlabels(spoke_labels)

    # add legend relative to top-left plotf"{index_names[i]}
    labels = (f"First: it. {index[0]}", f"Last: it. {index[1]}", f"Optimal: it. {index[2]}")
    legend = ax.legend(labels, loc=(0.9, .95),
                              labelspacing=0.1, fontsize='large')
    fig.show()

    return
