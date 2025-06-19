import pandas as pd
import os
import scanpy as sc

data_path = "../celloracle_data/files_for_basegrn_creation"
input_tsv_path = os.path.join(data_path, "tf_target_gene_list-4-28.tsv")
input_to_compare_old = os.path.join(data_path, "promoter_to_tf_list.csv")
output_csv_path = os.path.join(data_path, "reformatted_list-4-28.csv")
scrna_seq_data_dir ="../celloracle_data/scrna_final_celloc/V_unspliced_norm_log_W_MESC/trimmed/train_data.h5ad"

def parse_target_gene(identifier):
    """
    Parses the gene symbol from identifiers like 'P_542:Slc19a1:1'.
    Adjust this function if your identifier format is different.
    """
    try:
        parts = identifier.split(':')
    except AttributeError:
        print(identifier)
        print(f"Warning: Unexpected identifier format: {identifier}")
        return None
    if len(parts) >= 2:
        # Assuming the gene name is the second part
        return parts[1]
    else:
        print(identifier)
        print(f"Warning: Unexpected identifier format: {identifier}")
        return None

def create_csv_tf_to_gene():
    df = pd.read_csv(input_tsv_path, sep='\t',  skip_blank_lines=True, lineterminator='\n', header=0)
    print("head of the dataframe:", df.head())
    print("shape of the dataframe:", df.shape)
    #drop nans
    df.dropna(subset=['promoter', 'TF_inSwissRegulon'], inplace=True)

    print(f"Successfully read {len(df)} rows.")

    df['Target_Gene'] = df['promoter'].apply(parse_target_gene)

    df['TF'] = df['TF_inSwissRegulon'].astype(str).str.capitalize()

    # 3. Handle potential missing values after parsing/capitalization
    original_rows = len(df)
    df.dropna(subset=['TF', 'Target_Gene'], inplace=True)
    rows_after_dropna = len(df)


    grouped_targets = df.groupby('TF')['Target_Gene'].apply(lambda genes: ','.join(sorted(list(set(genes)))))

    output_df = grouped_targets.reset_index()

    output_df.rename(columns={'Target_Gene': 'Target_genes'}, inplace=True)
    print(output_df.head())

    output_df.to_csv(output_csv_path, index=False, sep='\t')

    compare_thing = pd.read_csv(input_to_compare_old, sep='\t', header=0)
    unique_tf_count = len(output_df)
    print(f"Found {unique_tf_count} unique TF entries after processing.")
    tfs_old  = compare_thing['TF_inSwissRegulon']
    tf_strings_series = tfs_old.astype(str)
    list_of_tf_lists = tf_strings_series.str.split(',')
    all_individual_tfs = list_of_tf_lists.explode()
    all_individual_tfs = all_individual_tfs.str.strip()
    all_individual_tfs = all_individual_tfs[all_individual_tfs != '']
    unique_tfs = all_individual_tfs.unique()
    unique_tf_count = len(unique_tfs)

    print(f"Total unique individual TF names found across all entries: {unique_tf_count}")

def create_dict_from_csv(adata:sc.AnnData) ->dict:
    df = pd.read_csv(output_csv_path, sep='\t', header=0)
    scrna_set = adata
    #create set of genes
    gene_set = set(scrna_set.var_names)
    print(f"Number of genes in the dataset: {len(gene_set)}")
    tf_to_target_dict = {}
    for index, row in df.iterrows():
        tf = row['TF']
        target_genes = row['Target_genes'].split(',')
        for gene in target_genes:
            if gene not in gene_set:
                continue
        tf_to_target_dict[tf] = target_genes
    return tf_to_target_dict

def gene_to_tf_list(adata:sc.AnnData) -> dict:
    """
    Create a dictionary mapping each target gene to its corresponding TFs.
    """
    df = pd.read_csv(output_csv_path, sep='\t', header=0)
    gene_to_tf_dict = {}
    for index, row in df.iterrows():
        tf = row['TF']
        target_genes = row['Target_genes'].split(',')
        for gene in target_genes:
            if gene not in gene_to_tf_dict:
                gene_to_tf_dict[gene] = []
            gene_to_tf_dict[gene].append(tf)
    return gene_to_tf_dict