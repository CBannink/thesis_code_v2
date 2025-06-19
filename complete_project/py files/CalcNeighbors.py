import logging
import os
import sys

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
from datetime import datetime

class Calc:

    def __init__(self, scrna_data, k_cell_state, k_umap):
        self.scrna_data = scrna_data
        self.k_cell_state = k_cell_state
        self.k_umap = k_umap

    def calc_neighbors(self):
        # Calculate neighbors
        sc.pp.neighbors(self.scrna_data, n_neighbors=self.k_cell_state, n_pcs=self.k_umap)
        return self.scrna_data
