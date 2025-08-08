import pickle


def merge_seeds_new(new_data, old_column_names):
    columns = list(new_data.columns)
    for column_name in old_column_names:
        columns.remove(column_name)
    # columns.remove('CPR_Measurement')
    # columns.remove('SOM_Measurement')
    columns.remove('seed')
    columns.remove('Measurement')

    grouped_df = new_data.groupby(columns)['Measurement'].mean().reset_index()
    grouped_df.rename(columns={'Measurement': 'Mean'}, inplace=True)
    grouped_df['Std'] = new_data.groupby(columns)['Measurement'].std().reset_index()['Measurement']

    return grouped_df

def get_sizes(dataframe):
    sizes = []
    list_n_robots = dataframe["NR"].unique()
    for n_robots in list_n_robots:
        list_n_herd = dataframe.loc[
            dataframe["NR"] == n_robots, "NH"
        ].unique()
        for n_herd in list_n_herd:
            sizes.append((n_robots, n_herd))

    return sizes

def metric_caged_and_in_path(CPR_perseed, SOM_perseed):
    new_data = CPR_perseed.copy()
    new_data.rename(columns={'Measurement': 'CPR_Measurement'}, inplace=True)
    new_data = new_data.join(SOM_perseed['Measurement'])
    new_data.rename(columns={'Measurement': 'SOM_Measurement'}, inplace=True)

    def l_function(row):
        if row['CPR_Measurement'] == 1 and row['SOM_Measurement'] == row['NH']:
            return 1
        else:
            return 0

    new_data['Measurement'] = new_data.apply(l_function, axis=1)

    return new_data

# EXPERIMENT_ID = "20230920-113107"

# som_ps = pickle.load(open(f"results/{EXPERIMENT_ID}/merged/SOM_perseed.pkl", "rb"))
# cpr_ps = pickle.load(open(f"results/{EXPERIMENT_ID}/merged/CPR_perseed.pkl", "rb"))

# new_data = metric_caged_and_in_path(cpr_ps, som_ps)
# print(new_data)

# ----------- DEBUG:

# df = new_data.drop(columns=['seed', 'Measurement', 'CPR_Measurement', 'SOM_Measurement'])
# df['Mean'] = grouped_df['Measurement']
# df['Std'] = new_data.groupby(columns)['Measurement'].std().reset_index()['Measurement']

# print(df)
# print(df)

# columns = list(df.columns)



# means = []
# stds = []
# for name, group in new_data.groupby(columns):
#     means.append(group['Measurement'].mean())
#     stds.append(group['Measurement'].std())
# df['Mean'] = means
# df['Std'] = stds
# print(df)

