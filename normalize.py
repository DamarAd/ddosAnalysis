import pandas as pd
import numpy as np

desired_width=320
pd.set_option('display.width', desired_width)
pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)

col_names = [
     'destination_port', 'flow_duration ', 'total_fwd_packets', 'total_bwd_packets', 'total_length_of_fwd_packets', 'total_length_of_bwd_packets',
    'fwd_packet_length_max', 'fwd_packet_length_min','fwd_packet_length_mean', 'fwd_packet_length_std',
    'bwd_packet_length_max', 'bwd_packet_length_min','bwd_packet_length_mean', 'bwd_packet_length_std',
    'flow_bytes', 'flow_packets', 'flow_iat_mean', 'flow_iat_std', 'flow_iat_max', 'flow_iat_min',
    'fwd_iat_total', 'fwd_iat_mean', 'fwd_iat_std', 'fwd_iat_max', 'fwd_iat_min', 'bwd_iat_total', 'bwd_iat_mean',
    'bwd_iat_std', 'bwd_iat_max', 'bwd_iat_min', 'fwd_psh_flags', 'bwd_psh_flags', 'fwd_urg_flags', 'bwd_urg_flags',
    'fwd_header_length', 'bwd_header_length', 'fwd_packets', 'bwd_packets', 'min_packet_length', 'max_packet_length',
    'packet_length_mean', 'packet_length_std', 'packet_length_variance', 'fin_flag_count', 'syn_flag_count',
    'rst_flag_count', 'psh_flag_count', 'ack_flag_count', 'urg_flag_count', 'cwe_flag_count', 'ece_flag_count',
    'down_up_ratio', 'average_packet_size', 'avg_fwd_segment_size', 'avg_bwd_segment_size', 'fwd_hd_length1', 'fwd_avg_bytes_bulk',
    'fwd_avg_packets_bulk', 'fwd_avg_bulk_rate', 'bwd_avg_bytes_bulk', 'bwd_avg_packets_bulk', 'bwd_avg_bulk_rate',
    'subflow_fwd_packets', 'subflow_fwd_bytes', 'subflow_bwd_packets', 'subflow_bwd_bytes', 'init_win_bytes_forward',
    'init_win_bytes_backward', 'act_data_pkt_fwd', 'min_seg_size_forward', 'active_mean', 'active_std', 'active_max',
    'active_min', 'idle_mean', 'idle_std', 'idle_max', 'idle_min', 'label'
]

dataframe = pd.read_csv("ddos_datatrain.csv", names=col_names, skiprows = 1, low_memory=False)

dataframe['flow_bytes'] = dataframe['flow_bytes'].astype(np.float64).replace(np.inf, np.nan).fillna(0)
dataframe['flow_packets'] = dataframe['flow_packets'].astype(np.float64).replace(np.inf, np.nan).fillna(0)

array = dataframe.values

# create fungsi array
def dis(arr):
    return pd.DataFrame(arr)

# Load data fitur - target
X = array[:,:-1]
Y = array[:,78]

# normalisasi atribut categoric
from sklearn.preprocessing import LabelEncoder
encode = LabelEncoder()
rescaledY =  encode.fit_transform(Y)

# normalisasi atribut numeric
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
rescaledX = scaler.fit_transform(X)

'''
print("SEBELUM NORMALISASI \n" , dis(X).head())
print("\n")
print("SESUDAH NORMALISASI \n" , dis(rescaledX).head())
'''

#Split data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectFromModel

# Random forest excec
rfc = RandomForestClassifier(n_estimators=1000, n_jobs=-1)
rfc.fit(X_train, Y_train)

# Print fitur beserta skor randomforest
print("SKOR RANDOM FOREST")
for feature in zip(col_names, rfc.feature_importances_):
    print(feature)

# Set threshold
sfm = SelectFromModel(rfc, threshold=0.005)
sfm.fit(X_train, Y_train)

# Print fitur yang penting
print("IMPORTANCE FITUR")
for selection in sfm.get_support(indices=True):
    print(col_names[selection])

# Transform data ke dataset baru yang telah diseleksi fiturnya
X_important_train = sfm.transform(X_train)
X_important_test = sfm.transform(X_test)

# random forest dengan importance fitur
rfc_important = RandomForestClassifier(n_estimators=1000, n_jobs=-1)

# Train klasifikasi baru dengan importance fitur
rfc_important.fit(X_important_train, Y_train)

from sklearn.metrics import accuracy_score

# Apply keselururhan fitur
Y_pred = rfc.predict(X_test)

# View akurasi seluruh fitur
full_feat = accuracy_score(Y_test, Y_pred)
print("FULL FEATURE ACCURATION")
print(full_feat)

# Apply keselurahan fitur
Y_important_pred = rfc_important.predict(X_important_test)

# View akurasi importance fitur dulu
importance_feat = accuracy_score(Y_test, Y_important_pred)
print("IMPORTANCE FEATURE ACCURATION")
print(importance_feat)

