import pickle
from CellOracle import celloracle as co
import os

Gen_Data_Dir =  '../celloracle_data'
CellOracle_Object_Dir_New = os.path.join(Gen_Data_Dir, 'celloracle_object/new_promoter_without_mescs_trimmed_test_own_umap')
oracle_path = os.path.join(CellOracle_Object_Dir_New, 'fit_cellOC.celloracle.oracle')
links_path = os.path.join(CellOracle_Object_Dir_New, 'filtered_links.celloracle.links')


oracle = co.load_hdf5(oracle_path)
links = co.load_hdf5(links_path)
oracle.get_cluster_specific_TFdict_from_Links(links)

with open(os.path.join(CellOracle_Object_Dir_New, 'ready_oracle.pkl'), 'wb') as f:
    pickle.dump(oracle, f, protocol=4)

