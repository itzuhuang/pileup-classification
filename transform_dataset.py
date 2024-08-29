import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import uproot as ur

def normalize(x):
    mean, std = np.mean(x), np.std(x)
    out = (x - mean) / std
    return out, mean, std

def apply_save_log(x):
    epsilon = 1e-10
    minimum = x.min()
    if x.min() <= 0:
        x = x - x.min() + epsilon
    else:
        minimum = 0
        epsilon = 0
    return np.log(x), minimum, epsilon

def getCustom(df):
    column_names = ['cluster_fracE','avgMu', 'nPrimVtx', 'nCluster', 'cluster_nCells', 'clusterE', 'clusterEta',
                    'cluster_sumCellE', 'cluster_CENTER_MAG', 'cluster_ENG_CALIB_TOT',
                    'cluster_CENTER_X', 'cluster_CENTER_Y', 'cluster_CENTER_Z',
                    'cluster_ENG_CALIB_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_TIME',
                    'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
                    'cluster_ISOLATION', 'clusterPhi', 'cluster_SIGNIFICANCE',
                    'cluster_CELL_SIGNIFICANCE', 'cluster_PTD', 'cluster_MASS']

    scales = {}
    field_names = ["clusterE", "cluster_CENTER_LAMBDA", "cluster_FIRST_ENG_DENS", "cluster_SECOND_TIME", "cluster_SIGNIFICANCE"]
    for field_name in field_names:
        x, minimum, epsilon = apply_save_log(df[field_name])
        x, mean, std = normalize(x)
        scales[field_name] = ("SaveLog / Normalize", minimum, epsilon, mean, std)
        df[field_name] = x

    field_names = ["clusterEta", "cluster_CENTER_MAG", "nPrimVtx", "avgMu"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        scales[field_name] = ("Normalize", mean, std)
        df[field_name] = x

    field_names = ["cluster_nCells","cluster_fracE","cluster_ENG_CALIB_FRAC_EM", "cluster_LATERAL", "cluster_LONGITUDINAL", "cluster_PTD", "cluster_ISOLATION"]
    for field_name in field_names:
        x = df[field_name]
        x, mean, std = normalize(x)
        scales[field_name] = ("Normalize", mean, std)
        df[field_name] = x
    return df

def sample_data(df, n_samples=250000):
    pileup = df[df['pile_up'] == 1].iloc[:n_samples]
    non_pileup = df[df['pile_up'] == 0].iloc[:n_samples]
    return pd.concat([pileup, non_pileup])


filename = "/data2/loch/ML4Calo/JetSamples_v3/rootfiles/Akt4EMTopo.inc.s100000000.v3.root"
file = ur.open(filename)
tree = file["ClusterTree"]
df = tree.arrays(library="pd")

def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi, dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi

def calculate_deltaR(df):
    delta_phi_values = delta_phi(df['truthPhi'], df['clusterPhi'])
    delta_eta_values = df['truthEta'] - df['clusterEta']
    deltaR = np.sqrt(delta_phi_values**2 + delta_eta_values**2)
    return deltaR

df['deltaR'] = calculate_deltaR(df)

df = df[(df["cluster_ENG_CALIB_TOT"] > 0) & (abs(df["clusterEta"]) < 2.4) &
        (df["clusterE"] > 0) & (df["cluster_FIRST_ENG_DENS"] > 0) & (df["deltaR"] < 0.4) &
        (df["cluster_CENTER_LAMBDA"] > 0) & (abs(df["clusterEta"] - df["clusterEtaCalib"]) < 0.1)]

#(df['jetCalPt'] < 80) &

df = df[df['jetCnt']%10==0]
print("Data points after initial filtering:", len(df))

#df['labels'] = np.where(df['cluster_ENG_CALIB_TOT'] >= 0.001, 1, 0)
df['response'] = df['clusterE'] / df['cluster_ENG_CALIB_TOT']
df['labels'] = np.where((df['response'] > 1.8) & (df['cluster_ENG_CALIB_TOT'] < 0.84), 1, 0)

column_names = ['cluster_fracE','avgMu', 'nPrimVtx', 'nCluster', 'cluster_nCells', 'clusterE', 'clusterEta',
                'cluster_sumCellE', 'cluster_CENTER_MAG', 'cluster_ENG_CALIB_TOT',
                'cluster_CENTER_X', 'cluster_CENTER_Y', 'cluster_CENTER_Z',
                'cluster_ENG_CALIB_FRAC_EM', 'cluster_FIRST_ENG_DENS', 'cluster_SECOND_TIME',
                'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
                'cluster_ISOLATION', 'clusterPhi', 'cluster_SIGNIFICANCE',
                'cluster_CELL_SIGNIFICANCE', 'cluster_PTD', 'cluster_MASS', 'labels', 'jetCnt',
                'eventNumber','jetRawE','jetRawPt','truthJetE','truthJetPt','jetCalE',
                'jetAreaE','jetAreaPt','clusterPt','clusterECalib']
df = df[column_names]
train_df = df[df['jetCnt'] % 100 <= 60]
test_df = df[(df['jetCnt'] % 100 == 70)|(df['jetCnt']==80)]
val_df = df[df['jetCnt'] % 100 == 90]

signal_df = df[df['labels']==1]
pu_df = df[df['labels']==0]

for i in range(10):
    print('Signal in jetCnt%100 == ',10*i,': ',len(signal_df[signal_df['jetCnt']%100==10*i]))
    print('Pile up in jenCnt%100 == ',10*i,': ',len(pu_df[pu_df['jetCnt']%100==10*i]))

train_df = train_df[(train_df['jetCnt']%100==0)|(train_df['labels']==0)]
train_sig = len(train_df[train_df['labels']==1])
train_pu = len(train_df[train_df['labels']==0])


print('Signal in train = ',train_sig)
print('Pile up in train = ',train_pu)
print('Fraction of pileup in train = ',train_pu/(train_pu + train_sig))
print('Size of train = ',len(train_df))
print('Size of test = ',len(test_df))
print('Size of val = ',len(val_df))

'''
train_df = sample_data(train_df)
test_df = sample_data(test_df)
val_df = sample_data(val_df)
'''


train_df.to_csv('untransformed_train-test.csv', index=False)
test_df.to_csv('untransformed_test-test.csv', index=False)
val_df.to_csv('untransformed_val-test.csv', index=False)

print('CSV files saved for untransformed data.')


t_df = getCustom(df)
pd.options.display.max_columns = None
print(t_df.columns.tolist())

t_train_df = t_df[t_df['jetCnt'] % 100 <= 60]
t_test_df = t_df[(t_df['jetCnt'] % 100 == 70)|(t_df['jetCnt']==80)]
t_val_df = t_df[t_df['jetCnt'] % 100 == 90]

t_train_df = t_train_df[(t_train_df['jetCnt']%100==0)|(t_train_df['labels']==0)]
t_train_sig = len(t_train_df[t_train_df['labels']==1])
t_train_pu = len(t_train_df[t_train_df['labels']==0])

print('Signal in train after transform = ',t_train_sig)
print('Pile up in train after transform = ',t_train_pu)
print('Fraction of pileup in train after transform = ',t_train_pu/(t_train_pu + t_train_sig))


'''
train_transformed = getCustom(train_df.drop(columns='pile_up').copy())
test_transformed = getCustom(test_df.drop(columns='pile_up').copy())
val_transformed = getCustom(val_df.drop(columns='pile_up').copy())


train_transformed['pile_up'] = train_df['pile_up'].values
test_transformed['pile_up'] = test_df['pile_up'].values
val_transformed['pile_up'] = val_df['pile_up'].values
'''
t_train_df.to_csv('transformed_train-test.csv', index=False)
t_test_df.to_csv('transformed_test-test.csv', index=False)
t_val_df.to_csv('transformed_val-test.csv', index=False)

print('CSV files saved for transformed data.')
exit()
def create_histograms(X, y, features, file_path):
    pdf_pages = PdfPages(file_path)
    positive_mask = y == 0
    negative_mask = y == 1
    num_bins = 30
    for feature in features:
        positive_data = X[positive_mask][feature]
        negative_data = X[negative_mask][feature]

        plt.figure(figsize=(10, 6))
        plt.hist(positive_data, bins=num_bins, alpha=0.5, label='No Pile Up', density=True, color='blue')
        plt.hist(negative_data, bins=num_bins, alpha=0.5, label='Pile Up', density=True, color='red')
        plt.title(f'Histogram of {feature}')
        plt.xlabel('Feature Value')
        plt.ylabel('Normalized Frequency')
        plt.legend()
        pdf_pages.savefig()
        plt.close()

    pdf_pages.close()

features = ['cluster_fracE','avgMu', 'nPrimVtx', 'nCluster', 'cluster_nCells', 'clusterE', 'clusterEta',
            'cluster_sumCellE', 'cluster_CENTER_MAG',
            'cluster_CENTER_X', 'cluster_CENTER_Y', 'cluster_CENTER_Z',
            'cluster_FIRST_ENG_DENS', 'cluster_SECOND_TIME',
            'cluster_CENTER_LAMBDA', 'cluster_LATERAL', 'cluster_LONGITUDINAL',
            'cluster_ISOLATION', 'clusterPhi', 'cluster_SIGNIFICANCE',
            'cluster_CELL_SIGNIFICANCE', 'cluster_PTD', 'cluster_MASS']

X_train_untransformed = train_df.drop(columns='pile_up')
y_train_untransformed = train_df['pile_up']
create_histograms(X_train_untransformed, y_train_untransformed, features, 'feature_histogram_untransformed_2c.pdf')

X_train_transformed = train_transformed.drop(columns='pile_up')
y_train_transformed = train_transformed['pile_up']
create_histograms(X_train_transformed, y_train_transformed, features, 'feature_histogram_transformed_2c.pdf')

print('Histograms saved for both untransformed and transformed data.')
