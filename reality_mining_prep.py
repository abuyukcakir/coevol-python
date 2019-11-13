import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd

from CoEVOL import CoEVOL

import sys
from datetime import datetime
from dateutil.parser import parse

from scipy.io import loadmat
from functools import reduce
from scipy.sparse import csr_matrix
# import matplotlib.pyplot as plt
# np.set_printoptions(threshold=sys.maxsize)

MONTHLY = 0
WEEKLY = 1

def preprocess(mode=MONTHLY):
    vc_date, vc_src, vc_dest, sm_date, sm_src, sm_dest, all_nodes = clean_data()
    max_id = np.max(all_nodes)
    print(len(vc_date))
    print(len(sm_date))

    if(mode == MONTHLY):
        # Parse dates month by month.
        vc_monthly_dates = get_monthly_dates(vc_date)
        sm_monthly_dates = get_monthly_dates(sm_date)

        # Generate monthly snapshots
        snapshots_vc = generate_snapshot_graphs(vc_monthly_dates, vc_src, vc_dest, max_id)
        snapshots_sm = generate_snapshot_graphs(sm_monthly_dates, sm_src, sm_dest, max_id)

        # Sanity check
        for i in range(len(snapshots_vc)):
            print(snapshots_vc[i].number_of_nodes())
            print(snapshots_vc[i].number_of_edges())

        return snapshots_vc, snapshots_sm, max_id

    elif(mode == WEEKLY):   # Parse dates week by week
        vc_weekly_dates = get_weekly_dates(vc_date)
        sm_weekly_dates = get_weekly_dates(sm_date)

        # Generate weekly snapshots
        snapshots_vc = generate_snapshot_graphs(vc_weekly_dates, vc_src, vc_dest, max_id)
        snapshots_sm = generate_snapshot_graphs(sm_weekly_dates, sm_src, sm_dest, max_id)

        # Sanity check
        # for i in range(len(snapshots_vc)):
        #     print(snapshots_vc[i].number_of_nodes())
        #     print(snapshots_vc[i].number_of_edges())

        return snapshots_vc, snapshots_sm, max_id
    else:
        print('Undefined date parsing mode. Returning.')
        return

def generate_snapshot_graphs(dates, src, dest, max_id):
    '''
    Given the date, source node, destination node arrays,
    generate snapshot graphs corresponding to each different date.
    '''
    cur_snapshot = create_empty_snapshot(max_id)
    cur_date = dates[0]
    snapshot_graphs = []

    for i in range(len(dates)):
        if(dates[i] != cur_date): # len(dates[i]) > 0 and
            # We are done with the current snapshot.
            # Add that to the list.
            snapshot_graphs.append(cur_snapshot)
            # print('new snapshot is generated')

            # Generate a new snapshot graph.
            cur_snapshot = create_empty_snapshot(max_id)
        # Otherwise, add the i-th (temporal) edge to the current snapshot
        cur_snapshot.add_edge(src[i], dest[i])
        cur_date = dates[i]


    return snapshot_graphs

def create_empty_snapshot(max_id):
    g = nx.Graph()
    node_ids = np.arange(0, max_id+1)
    g.add_nodes_from(node_ids)

    return g

def clean_data():
    dates_sm = loadmat('RealityMining/Date_Short_msg.mat')
    src_sm = loadmat('RealityMining/Source_Short_msg.mat')
    dest_sm = loadmat('RealityMining/Destination_Short_msg.mat')

    dates_vc = loadmat('RealityMining/Date_Voice_Call.mat')
    src_vc = loadmat('RealityMining/Source_Voice_Call.mat')
    dest_vc = loadmat('RealityMining/Destination_Voice_Call.mat')

    # dates_bt = loadmat('RealityMining/Date_text_Bluetoothe.mat')
    # src_bt = loadmat('RealityMining/Source_realitymining_Bluetoothe.mat')
    # dest_bt = loadmat('RealityMining/Destination_realitymining_Bluetoothe.mat')

    vc_date = np.stack(dates_vc['Date'], axis=1)[0]
    vc_src = np.stack(src_vc['Source'], axis=1)[0]
    vc_dest = np.stack(dest_vc['Destination'], axis=1)[0]

    sm_date = np.stack(dates_sm['Date'], axis=1)[0]
    sm_src = np.stack(src_sm['Source'], axis=1)[0]
    sm_dest = np.stack(dest_sm['Destination'], axis=1)[0]

    # bt_date = np.stack(dates_vc['Bluetoothe'], axis=1)[0]

    all_nodes = reduce( np.union1d, (vc_src, vc_dest, sm_src, sm_dest))

    return vc_date, vc_src, vc_dest, sm_date, sm_src, sm_dest, all_nodes

def get_monthly_dates(date_arr):
    '''
    Reads Date arrays of RealityMining datasets and removes day information
    from the dates. The resulting format is Mon-YYYY.
    '''
    monthly = np.empty_like(date_arr)

    for i in range(len(date_arr)):
        # print(date_arr[i][0])
        # i-th entry is an array of a single element. Take that one and
        # remove the Day- entry from the head of the date.
        monthly[i] = date_arr[i][0][3:]

    return monthly

def get_weekly_dates(date_arr):
    '''
    Reads Date arrays of RealityMining datasets and changes each day by
    WeekNumber-YYYY where WeekNumber is which week of the year is that week.
    '''
    weekly = np.empty_like(date_arr)
    for i in range(len(date_arr)):
        cur_date = datetime.strptime(date_arr[i][0], '%d-%b-%Y')
        y, wn, wd = cur_date.isocalendar()

        weekly[i] = '{}-{}'.format(wn, date_arr[i][0][7:])

    # print(weekly)
    return weekly

if(__name__ == "__main__"):
    snapshots_m_vc, snapshots_m_sm, max_nodeid = preprocess(mode=MONTHLY)
    # snapshots_w_vc, snapshots_w_sm = preprocess(mode=WEEKLY)

    # Number of timestamps T
    T = len(snapshots_m_vc)
    # Number of subjects
    S = 2
    # Number of nodes in one snapshot
    n = max_nodeid+1

    A = np.empty((T, S), dtype=object)

    # Read graph snapshots as matrices, and input them to CoEvol
    for i in range(len(snapshots_m_vc)):
        A[i, 0] = nx.to_scipy_sparse_matrix(snapshots_m_vc[i])

    for i in range(len(snapshots_m_sm)):
        A[i, 1] = nx.to_scipy_sparse_matrix(snapshots_m_sm[i])

    print('Reality Mining dataset is read.')
    print('Running CoEVOL on the dataset...')

    # ks = [5, 10, 15, 20, 25]
    thetas = [0.1, 0.3, 0.5, 0.7, 0.9]

    # thetas = [0.1]
    ks = [5, 10]

    errs = np.zeros((len(ks), len(thetas)))
    for i in range(len(ks)):
        for j in range(len(thetas)):
            coevol = CoEVOL(A, k=ks[i], theta=thetas[j])
            coevol.factorize()
