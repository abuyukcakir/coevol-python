import networkx as nx
import numpy as np
import scipy as sp
import pandas as pd
import argparse
import matplotlib.pyplot as plt
from networkx.readwrite.edgelist import read_edgelist
from graph_util import draw_degree_histogram

def parse_args():
    parser = argparse.ArgumentParser(description="CoEVOL as described in Yu et al. 2018 \
    Modeling Co-Evolution Across Multiple Networks.")

    parser.add_argument('--input', nargs='?', default='/data/test.tsv', help='Input graph file path')

    parser.add_argument('--nlatent', type=int, default=10, help='Latent dimension size of the decomposition.')

    parser.add_argument('--theta', type=float, default=0.3, help='Parameter of the exponential decay function.')

    parser.add_argument('--alpha', type=float, default=0.01, help='Regularization parameter on U')

    parser.add_argument('--beta', type=float, default=0.01, help='Regularization parameter on X')

    parser.add_argument('--gamma', type=float, default=0.01, help='Regularization parameter on Y')

    return parser.parse_args()

def weighted_common_neighbors():
    return

def weighted_adamic_adar():
    return

def high_performance_link_prediction():
    return

def nonnegative_matrix_factorization():
    return

def cp_tensor_factorization():
    return


if(__name__ == '__main__'):
    args = parse_args()

    input_file = args.input
    k = args.nlatent
    theta = args.theta
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma

    print('------------------------')
    print('Input file: \t {}'.format(input_file))
    print('Latent Dimension Size: \t {}'.format(k))
    print('Theta (Decay parameter): \t {}'.format(theta))
    print('Alpha (Reg. on U): \t {}'.format(alpha))
    print('Beta (Reg. on X): \t {}'.format(beta))
    print('Gamma (Reg. on Y):: \t {}'.format(gamma))

    g = read_edgelist(input_file, nodetype=int)
    print( nx.number_of_edges(g) )
    print( nx.number_of_nodes(g) )
    draw_degree_histogram(g)
