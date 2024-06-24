'''
Script for preprocessing data
'''

import numpy as np
import pandas as pd

# from math import sqrt, sin, cos, pi, asin


def common_part_of_commuters(values1, values2, numerator_only=False):
    """
    Evaluate similarity between the generated flows and the real flows.

    :param values1: list/series of the predicted flows
    :param values2: list/series of the real flows
    """
    if numerator_only:
        tot = 1.0
    else:
        tot = (np.sum(values1) + np.sum(values2))
    if tot > 0:
        return 2.0 * np.sum(np.minimum(values1, values2)) / tot
    else:
        return 0.0


def visit_to_flow(visit_data):
    """
    Extract flows from the data of port visits

    :param visit_data: pandas dataframe of port visits
    """
    visit_arr = visit_data.sort_values(by=['mmsi','start']).to_numpy()

    ## The above sorted dataset should be converted to a format with "source", "target"
    ## then use the function `from_pandas_edgelist()`
    mmsi = None
    staying_end = None # The end time point of a ship stay at a port
    mmsi_list = []
    source_port = []
    target_port = []
    source_lat = []
    source_lon = []
    target_lat = []
    target_lon = []
    source_time = []
    target_time = []
    ship_type = []
    ship_type_txt = []

    for obs in visit_arr:
        if obs[1] != mmsi:
            source_port.append(np.nan)
            source_time.append(np.nan)
            source_lat.append(np.nan)
            source_lon.append(np.nan)
            mmsi = obs[1]
        else:
            source_port.append(target_port[-1])
            source_time.append(staying_end)
            source_lat.append(target_lat[-1])
            source_lon.append(target_lon[-1])
        
        # Appending the current record (row in the array) to targeting info
        target_port.append(obs[0]) # 'port'
        target_time.append(obs[4]) # 'start'
        target_lat.append(obs[12])
        target_lon.append(obs[13])
        mmsi_list.append(mmsi)
        ship_type.append(obs[6])
        ship_type_txt.append(obs[10])
        staying_end = obs[5] # 'end' time

    # Creat the dataframe
    edge_dict = {'mmsi': mmsi_list,
                'source_port': source_port,
                'target_port': target_port,
                'source_lat': source_lat,
                'source_lon': source_lon,
                'target_lat': target_lat,
                'target_lon': target_lon,
                'source_time': source_time,
                'target_time': target_time,
                'ship_type': ship_type,
                'ship_type_txt': ship_type_txt}

    edge_df = pd.DataFrame(edge_dict)
    edge_df.dropna(subset=['source_port'], inplace=True)
    # edge_df['time_interval'] = edge_df['target_time'] - edge_df['source_time']
    edge_df = edge_df.astype({'source_port': int})
    # edge_df['time_interval'] = edge_df.time_interval.dt.days

    # # Adding source and target month according to the time info
    # edge_df['source_month'] = edge_df['source_time'].dt.month
    # edge_df['target_month'] = edge_df['target_time'].dt.month

    # Removing the records without movement (showing the same source and target port)
    edge_df = edge_df.loc[edge_df.source_port!=edge_df.target_port].reset_index(drop=True) # 93.7% of the records left
    
    return edge_df

def earth_distance(lat1, lng1, lat2, lng2):
    dlat, dlng = lat1-lat2, lng1-lng2
    # ds = 2 * asin(sqrt(sin(dlat/2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlng/2.0) ** 2))
    # Rewriting with numpy to enable broadcasting
    ds = 2 * np.arcsin(np.sqrt(np.sin(dlat/2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2.0) ** 2))
    return 6371.01 * ds  # spherical earth...


def get_population(flux, port_id=None):
    if port_id:
        return flux.loc[flux['port']==port_id,'NumberOfStays'].iloc[0]
    else:
        raise Exception("Unable to return a population since port ID does not exist.")


def get_distance(port_i, port_j, flux):
    lat1 = flux[flux.port == port_i].latitude.values
    lon1 = flux[flux.port == port_i].longitude.values
    lat2 = flux[flux.port == port_j].latitude.values
    lon2 = flux[flux.port == port_j].longitude.values
    return earth_distance(lat1, lon1, lat2, lon2)

def get_export_value(cc_i, cc_j, export):
    """
    :param cc_i: country code of the source port
    :param cc_j: country code of the destination port
    :param export: dataframe containing the export value in dollars from country i to j
    """
    exp_val = export[(export['location_code']==cc_i) & (export['partner_code']==cc_j)].export_value
    if exp_val.empty:
        return 0.0
    else:
        return exp_val.to_numpy()[0]

def get_features_gravity(origin, destination, df='exponential'):
    distance = earth_distance(origin.location, destination.location)
    return [origin.outflow] + [destination.outflow] + [distance]

def get_target_proba(source_port_id, flow_size, df_outflow):
    tot_outflow_source = df_outflow.loc[df_outflow['source_port']==source_port_id, 'outflow'].iloc[0]
    return flow_size / tot_outflow_source