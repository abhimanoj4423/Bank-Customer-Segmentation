import pandas as pd
import numpy as np
import streamlit as st
import joblib

from streamlit_option_menu import option_menu
from sklearn.impute import *
from sklearn.preprocessing import *
from sklearn.feature_selection import *

from sklearn.decomposition import PCA
from sklearn.decomposition import *
from sklearn.manifold import TSNE

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

Key = st.secrets["Key"]
Id = st.secrets["Id"]

st.set_page_config(
    layout="wide",
    initial_sidebar_state="expanded")

cl_model = joblib.load(open('rfclassifier.joblib', 'rb'))
cu_model = joblib.load(open('kmeans.joblib', 'rb'))
imp_median = joblib.load(open('imp_median.joblib', 'rb'))
imp_mode = joblib.load(open('imp_mode.joblib', 'rb'))
oe_edu = joblib.load(open('oe_edu.joblib', 'rb'))
oe_job = joblib.load(open('oe_job.joblib', 'rb'))

std =StandardScaler()

label_en = LabelEncoder()
ohe = OneHotEncoder(sparse_output=False, drop='first')

# Function to tranform the data and to predict the outcomes of test data
def classification_predict(df):
    column_transformer = ColumnTransformer(transformers=[
    ("onehot", ohe, ['communication_type','prev_campaign_outcome']),
    ("standard", std, ['customer_age', 'job_type', 'month', 'last_contact_duration','num_contacts_prev_campaign'])])

    pipeline = Pipeline([('preprocessing', column_transformer)])

    transformed_data = pd.DataFrame(pipeline.fit_transform(df),columns=cl_model.feature_names_in_)
    y_pred = pd.DataFrame(cl_model.predict(transformed_data),columns=['term_deposit_subscribed'])
    pred = pd.concat([df, y_pred], axis=1)
    return pred

def encode_categorical_columns(df):
    df_encoded = df.copy()  # Make a copy of the DataFrame
    label_encoders = {}
    one_hot_encoders = {}
    for column in df_encoded.select_dtypes(include=['object', 'category']).columns:
        unique_values = df_encoded[column].nunique()

        if unique_values == 2:
            le = LabelEncoder()
            df_encoded[column] = le.fit_transform(df_encoded[column])
            label_encoders[column] = le

        else:
            ohe = OneHotEncoder(sparse_output=False, drop='first')
            ohe_df = pd.DataFrame(ohe.fit_transform(df_encoded[[column]]), columns=ohe.get_feature_names_out([column]),
                                  index=df_encoded.index)
            df_encoded = df_encoded.drop(column, axis=1).join(ohe_df)
            one_hot_encoders[column] = ohe
    return df_encoded

Simple_Imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')

def imputer(encoded_data,Imputer):
    if Imputer is not None:
        imputer_df = Imputer.fit_transform(encoded_data)
        imputer_df = pd.DataFrame(imputer_df,columns=encoded_data.columns)
    else:
        imputer_df = encoded_data.copy()

    return imputer_df

scalers = {'standard': StandardScaler(),'normalize': Normalizer(),'minmax': MinMaxScaler()}

def scale_dataframe(df,scaling_type):
    scaled_df = df.copy()
    if scaling_type == 'None':
        return scaled_df

    elif scaling_type in scalers:
        columns_to_scale = [col for col in scaled_df.columns if scaled_df[col].nunique() > 3]
        stype = scalers[scaling_type]
        scaled_df[columns_to_scale] = stype.fit_transform(scaled_df[columns_to_scale])

    return scaled_df

fr={"T-distributed Stochastic Neighbor Embedding":TSNE,"Kernel Principal component analysis":KernelPCA,"Principal component analysis":PCA,
    "Mini-batch Sparse Principal Components Analysis":MiniBatchSparsePCA}

def feature_reduction(scaled_data,n_comp,tech):
    unsup_col=[]
    for i in range(0,n_comp):
        col='PCA'+str(i+1)
        unsup_col.append(col)
    dim_reducer=fr[tech]
    reducer = dim_reducer(n_components=n_comp)
    fs_df = reducer.fit_transform(scaled_data)
    fs_df = pd.DataFrame(fs_df, columns=unsup_col)
    return fs_df

month_map = {
    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4, 'may': 5, 'jun': 6,
    'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
}

education_map = { 'unknown':0, 'primary':1, 'secondary':2,'tertiary':3}
job_types_map = {'unknown':0, 'retired':1, 'unemployed':2, 'student':3, 'housemaid':4, 'blue-collar':5,
               'services':6, 'admin.':7, 'technician':8, 'self-employed':9, 'entrepreneur':10, 'management':11}

balance_bins = [-8020, 0, 1000,5000, 10000, 20000, 102128]
balance_labels = ['-10000-0', '0-1000', '1001-5000','5001-10000', '10001-20000', '20001+']

bins = [18, 24, 35, 50, 65, 100]
labels = ['18-24', '25-35', '36-50', '51-65', '66+']

with st.sidebar:
    st.title('__Customer Churn Predictor__')
    choice = option_menu(menu_title=None,
                         options=["Subcription Prediction","Customer Segmentation"])
    
if os.path.exists("cl_predicted.csv"):
    data = pd.read_csv('cl_predicted.csv', index_col=None)

if choice == 'Subcription Prediction':
    st.title("Term Deposit Subscription Prediction")

    st.subheader('Import the Dataset',divider='rainbow')

    data = st.file_uploader(" ")
    st.info("__Note:__ The test dataset should NOT contain __'term deposit subscribed'__ column")

    if data:
        data = pd.read_csv(data, index_col=None)
        st.success('File Upload Successfully')

        with st.expander("View Original Test DataFrame"):
            st.dataframe(data)
            st.text(f'Shape: {data.shape}')       

        columns=['customer_age', 'housing_loan', 'communication_type', 'month', 'last_contact_duration', 'num_contacts_prev_campaign', 
                    'prev_campaign_outcome', 'AgeGroup']
        
        data = imputer(data,Simple_Imputer)

        data['AgeGroup'] = pd.cut(data['customer_age'], bins=bins, labels=labels, right=False)
        data['BalanceGroup'] = pd.cut(data['balance'], bins=balance_bins, labels=balance_labels, right=False)

        data1 = data[columns]

        data1['month'] = data1['month'].replace(month_map)

        data1["customer_age"] = pd.to_numeric(data1.customer_age, errors='coerce')
        data1["month"] = pd.to_numeric(data1.month, errors='coerce')
        data1["last_contact_duration"] = pd.to_numeric(data1.last_contact_duration, errors='coerce')
        data1["num_contacts_prev_campaign"] = pd.to_numeric(data1.num_contacts_prev_campaign, errors='coerce')

        data1["AgeGroup"]=data1["AgeGroup"].astype('category')

        encoded_df = encode_categorical_columns(data1)

        scaled_df = scale_dataframe(encoded_df,'standard')

        with st.expander("View Preprocessed Test DataFrame"):
            st.dataframe(scaled_df)
            st.text(f'Shape: {scaled_df.shape}')

        if st.button("Predict",type='primary'):
            st.subheader("The Predicted Dataframe")
            st.info("Note: The Predicted column is added to the original Dataframe rather than the Preprocessed Data")
            y_pred = cl_model.predict(scaled_df)
            y_pred = pd.DataFrame(y_pred)
            data['term_deposit_subscribed'] = y_pred
            st.dataframe(data)
            st.download_button("Download Dataset", data.to_csv(), "cl_predicted.csv")

if os.path.exists("cu_predicted.csv"):
    test_cl = pd.read_csv('cu_predicted.csv', index_col=None)

if choice == 'Customer Segmentation':
    st.title("Customer Segmentation using Clustering")
    st.subheader('Import the Dataset',divider='rainbow')

    test_cl = st.file_uploader(" ")
    st.error("__Note:__ __'Remove'__ the above DataFrame and import the test dataset containing __'term deposit subscribed'__ column")

    if test_cl:
        test_cl = pd.read_csv(test_cl, index_col=None)
        st.success('File Upload Successfully')

        with st.expander("View Original Test DataFrame"):
            st.dataframe(test_cl)
            st.text(f'Shape: {test_cl.shape}')  
        
        cols=['customer_age', 'job_type', 'education', 'default','balance','housing_loan','personal_loan','term_deposit_subscribed']
        
        test_cl = imputer(test_cl,Simple_Imputer)

        test_cl['AgeGroup'] = pd.cut(test_cl['customer_age'], bins=bins, labels=labels, right=False)
        test_cl['BalanceGroup'] = pd.cut(test_cl['balance'], bins=balance_bins, labels=balance_labels, right=False)

        test1 = test_cl[cols]

        test1["customer_age"] = pd.to_numeric(test1.customer_age, errors='coerce')
        test1["balance"] = pd.to_numeric(test1.balance, errors='coerce')

        test1['job_type'] = oe_job.transform(test1[['job_type']])
        test1['education'] = oe_edu.transform(test1[['education']])

        encoded_df = encode_categorical_columns(test1)

        scaled_df = scale_dataframe(encoded_df,'standard')

        with st.expander("View Preprocessed Test DataFrame"):
            st.dataframe(scaled_df)
            st.text(f'Shape: {scaled_df.shape}')

        pca_df=feature_reduction(scaled_df,2,"Principal component analysis")

        with st.expander("View Feature Reduced Test DataFrame"):
            st.dataframe(pca_df)
            st.text(f'Shape: {pca_df.shape}')

        if st.button("Predict",type='primary'):
            st.subheader("The Predicted Dataframe")
            st.info("Note: The Predicted column is added to the original Dataframe rather than the Preprocessed Data")
            clusters = cu_model.predict(pca_df)

            #clusters = pd.DataFrame({"Customers": test.index, "Clusters": clusters})
            test_cl["Cluster_no"] = clusters

            st.dataframe(test_cl)
            st.download_button("Download Dataset", test_cl.to_csv(), "cu_predicted.csv")
