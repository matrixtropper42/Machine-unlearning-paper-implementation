import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import dill
import os
import zipfile
import requests
import streamlit as st

torch.classes.__path__ = []
def main():
    # Page configuration
    st.set_page_config(
        page_title="Machine Unlearning with SISA",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    # Github repo details
    owner = "WALKCART"
    repo = "Machine-Unlearning-Implementation"

    # Release tags
    release_tags = 'Files'
    # Fetch the token from environment variables
    token = st.secrets["GITHUB_TOKEN"] or os.getenv("GITHUB_TOKEN")

    # Check if the token is present
    if not token:
        raise ValueError("GitHub token not found. Make sure it's set in your environment variables.")

    # Authentication header
    headers = {
        "Authorization": f"token {token}",
        "User-Agent": "python-requests"
    }

    # Folder where the files will be saved
    download_dir = 'Data_and_Files'

    # Create the directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    download_assets(download_dir, release_tags, owner, headers, repo) # download the assets

    # loading the data from the different files
    shard_models = load_shard_models()
    X_test = load_X_test()
    X_train = load_X_train()
    y_test = load_y_test()
    y_train = load_y_train()
    test_shards = load_test_shrds()
    train_shards = load_train_shrds()
    test_slices = load_test_sls()
    train_slices = load_train_sls()
    df = load_df()
    scaler = load_scaler()
    num_shards = 30
    num_slices = 25


    if "training_rmses" not in st.session_state:
        st.session_state.training_rmses = []
    if "testing_rmses" not in st.session_state:
        st.session_state.testing_rmses = []

    st.title('Machine Unlearning with SISA Training Approach')
    st.divider()
    st.write('\n')
    st.write('\n')
    st.subheader('Challenges in Machine Unlearning')
    st.write('Companies collect vast amounts of user data to train machine learning models, but rising privacy concerns and regulations like GDPR and CCPA now require them to delete not just user data but also its influence on models. Since models often memorize aspects of training data, this poses a challenge. Traditional unlearning, which involves retraining models from scratch after removing specific data, is highly inefficient due to the sheer size of datasets, number of features, and model complexity. This inefficiency underscores the need for faster, more scalable unlearning techniques.')
    st.write('\n')
    st.write('\n')
    st.divider()
    st.subheader('SISA Training Approach')
    st.markdown(
    """
    **Sharded, Isolated, Sliced, Aggregated (SISA) training** is an innovative method designed to efficiently 
    address the challenges of unlearning specific data points in machine learning. This approach, detailed in 
    [this research paper](https://arxiv.org/pdf/1912.03817), works as follows:
    
    1. **Sharding:**  
       The dataset is divided into multiple distinct shards.
    
    2. **Isolation:**  
       Each shard is completely independent, ensuring no data points are shared between shards.
    
    3. **Slicing:**  
       Each shard is further split into smaller slices to accelerate the unlearning process.
    
    4. **Aggregation:**  
       Each shard is associated with its own model.  
       - For **regression**, the final output is the **mean** or **median** of all model predictions.  
       - For **classification**, the final decision is based on **majority voting** or a similar method.
    """
    )
    st.divider()

    st.subheader("Learning and Unlearning Process in SISA Training")
    st.markdown(
        """
        The **learning and unlearning process** in SISA training works as follows:
        
        1. **Learning Process:**  
        - The model is initially trained on the **first slice** of a shard (e.g., the *kth shard*), and its parameters are saved.  
        - Next, it is trained on the **union of slices 1 and 2**, and the updated parameters are stored.  
        - This process continues slice by slice, saving parameters at each step.
        
        2. **Unlearning Process:**  
        - When an unlearning request is made, the **shard** and **slice** containing the data point are identified.  
        - The model’s parameters are reinitialized to the state just before the identified slice.  
        - The model is then retrained from that slice onward, following the same slice-wise learning approach.
        """
    )
    st.divider()
    
    st.subheader("Dataset for SISA Implementation")

    st.markdown(
        """
        The dataset used for implementing the SISA approach is the 
        [Fannie Mae and Freddie Mac Loan-Level Dataset](https://www.kaggle.com/datasets/thedevastator/2016-fannie-mae-and-freddie-mac-loan-level-datas) from Kaggle.
        
        - **Sample Size:** Approximately 3.5% of the original dataset (~88k data points) was randomly sampled for training.  
        - **Target Variable:** Acquisition Unpaid Principal Balance (UPB).  
        - **Features Used:**  
            - Record Number  
            - Borrower’s Annual Income  
            - Area Median Family Income (2016)  
            - Tract Income Ratio  
            - 2010 Census Tract - Median Income  
            - Purpose of Loan  
            - Number of Borrowers  
            - First-Time Home Buyer  
            - Federal Guarantee  
            - Property Type  
            - Rate Spread  
            - Occupancy Code  
            - US Postal State Code  
            - Metropolitan Statistical Area (MSA) Code  
            - Lien Status
        """
    )
    st.divider()
    st.subheader('Submit Unlearning Requests for Model Retraining')
    st.markdown(
    """
    This app allows you to:
    - **Locate a data point for deletion**
    - **Unlearn the model by removing its influence**
    """
    )
    # calculating rmse at the beginning and storing for graphing
    initial_train_rmse, initial_test_rmse = calc_rmse(shard_models, X_train, y_train, X_test, y_test)
    if not st.session_state.training_rmses:
        st.session_state.training_rmses.append(initial_train_rmse)
        st.session_state.testing_rmses.append(initial_test_rmse)
    st.write('\n')

    st.write('Use the following table showing the records in the data sampled for unlearning requests')
    st.write(df['Record Number'])
    st.subheader("Step 1: Enter Data for Deletion")
    record_no = st.number_input("Enter the Record Number to delete:", min_value=0, step=1)
    if 'test_or_train' not in st.session_state:
        st.session_state.test_or_train = None
        st.session_state.shard_idx_todel = None
        st.session_state.slice_idx_todel = None
        st.session_state.idx_inslice_todel = None
        st.session_state.idx_indf_todel = None
    if st.button("Find Data Point"):
        with st.spinner('Finding Data'):
            indices = find_shard_slice(record_no, df, scaler, num_shards, num_slices, train_slices, test_slices, X_train, X_test)
        if indices != None:
            test_or_train, shard_idx_todel, slice_idx_todel, idx_inslice_todel, idx_indf_todel = indices
            st.session_state.test_or_train = test_or_train  # Save to session_state
            st.session_state.shard_idx_todel = shard_idx_todel
            st.session_state.slice_idx_todel = slice_idx_todel
            st.session_state.idx_inslice_todel = idx_inslice_todel
            st.session_state.idx_indf_todel = idx_indf_todel 
            if test_or_train == 0:
                st.success(f"The data was found in the training set at {st.session_state.shard_idx_todel} indexed shard, {st.session_state.slice_idx_todel} indexed slice and at index {st.session_state.idx_inslice_todel} in the slice. The data was at {st.session_state.idx_indf_todel} in the main dataset")
            else:
                 st.success(f"The data was found in the testing set at {st.session_state.shard_idx_todel} indexed shard, {st.session_state.slice_idx_todel} indexed slice and at index {st.session_state.idx_inslice_todel} in the slice. The data was at {st.session_state.idx_indf_todel} in the main dataset")
        else:
            st.write('The record number was not in the dataset')
        
    st.write()
    if st.button('Delete Data and Unlearn'):
        with st.spinner('Deleting Data'):
            del_data(record_no, st.session_state.test_or_train, st.session_state.shard_idx_todel, st.session_state.slice_idx_todel, st.session_state.idx_inslice_todel, st.session_state.idx_indf_todel, test_slices, train_slices, df, X_test, X_train, y_test, y_train)
        st.success(f'Data Deleted successfully at shard index {st.session_state.shard_idx_todel}, slice {st.session_state.slice_idx_todel}, index in slice {st.session_state.idx_inslice_todel} and index in main dataset at {st.session_state.idx_indf_todel}')

        unlearn(st.session_state.test_or_train, num_slices, st.session_state.shard_idx_todel, st.session_state.slice_idx_todel, train_slices, shard_models, 32, 100)

        with st.spinner('Calculating RMSE'):
            train_rmse, test_rmse = calc_rmse(shard_models, X_train, y_train, X_test, y_test)
        st.success('RMSE Calculated')
        st.session_state.training_rmses.append(train_rmse)
        st.session_state.testing_rmses.append(test_rmse)

        # TRAINING RMSES
        with st.spinner('Graphing RMSE'):
            fig_training = go.Figure()
            fig_training.add_trace(go.Scatter(
                y=st.session_state.training_rmses, 
                mode='lines+markers', 
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                name='Data Line'
            ))
            fig_training.update_layout(
                title="Training RMSES",
                xaxis_title="Unlearning Cycle",
                yaxis_title="Training RMSE",
                template="plotly_dark", 
            )
            st.plotly_chart(fig_training)

            # TESTING RMSES PLOT
            fig_testing = go.Figure()
            fig_testing.add_trace(go.Scatter(
                y=st.session_state.testing_rmses, 
                mode='lines+markers', 
                line=dict(color='blue', width=2),
                marker=dict(size=8),
                name='Data Line'
            ))
            fig_testing.update_layout(
                title="Testing RMSES",
                xaxis_title="Unlearning Cycle",
                yaxis_title="Testing RMSE",
                template="plotly_dark",  
            )
            st.plotly_chart(fig_testing) 
        st.success('Graphing Completed!')


def download_assets(download_dir, release_tag, owner, headers, repo):
    # Ensure the download directory exists
    os.makedirs(download_dir, exist_ok=True)

    # Updated list of expected file names
    expected_files = [
        "shard_models.mdl", "X_test.dat", "X_train.dat", "y_test.dat", "y_train.dat",
        "test_shards_npy.shrds", "train_shards_npy.shrds", 
        "test_slices_npy.sls", "train_slices_npy.sls", "scaler.scl"
    ]

    # Check if all files are already downloaded
    files_downloaded = all(os.path.exists(os.path.join(download_dir, file_name)) for file_name in expected_files)

    if not files_downloaded:
        url = f"https://api.github.com/repos/{owner}/{repo}/releases/tags/{release_tag}"
        response = requests.get(url, headers=headers, timeout=30)

        if response.status_code == 200:
            try:
                release_data = response.json()
                if "assets" in release_data:
                    for asset in release_data["assets"]:
                        asset_name = asset["name"]
                        asset_url = asset["browser_download_url"]
                        asset_path = os.path.join(download_dir, asset_name)

                        # Skip if file exists
                        if os.path.exists(asset_path):
                            continue

                        with requests.get(asset_url, headers=headers, stream=True, timeout=60) as file_response:
                            if file_response.status_code == 200:
                                file_size = int(file_response.headers.get('content-length', 0))
                                chunk_size = 1024 * 1024 * 5  # 5 MB
                                downloaded = 0
                                progress_bar = st.progress(0)
                                status_text = st.empty()

                                with open(asset_path, "wb") as f:
                                    for chunk in file_response.iter_content(chunk_size=chunk_size):
                                        if chunk:
                                            f.write(chunk)
                                            downloaded += len(chunk)
                                            progress = downloaded / file_size
                                            progress_bar.progress(progress)
                                            status_text.text(f"Downloading {asset_name}: {progress * 100:.2f}%")

                                st.success(f"Downloaded: {asset_name}")
                            else:
                                st.error(f"Failed to download {asset_name}")
                else:
                    st.warning(f"No assets found in release {release_tag}")
            except ValueError as e:
                st.error(f"Error decoding JSON: {e}")
        else:
            st.error(f"Failed to fetch release data for {release_tag}: {response.status_code}")
            st.text(response.text)
    else:
        st.success("All files are already downloaded.")

@st.cache_resource
def load_shard_models():
    # opening the shard_models
    with open('Data_and_Files/shard_models.mdl', 'rb') as fh:
        shard_models = dill.load(fh)
        fh.close()
    return shard_models
        

# opeining the training and the testing data sets
@st.cache_resource
def load_X_test():
    with open('Data_and_Files/X_test.dat', 'rb') as fh:
        X_test = dill.load(fh)
        fh.close()
    return X_test
    
@st.cache_resource
def load_X_train():
    with open('Data_and_Files/X_train.dat', 'rb') as fh:
        X_train = dill.load(fh)
        fh.close()
    return X_train

@st.cache_resource
def load_y_test():
    with open('Data_and_Files/y_test.dat', 'rb') as fh:
        y_test = dill.load(fh)
        fh.close()
    return y_test

@st.cache_resource
def load_y_train():
    with open('Data_and_Files/y_train.dat', 'rb') as fh:
        y_train = dill.load(fh)  
        fh.close()
    return y_train

@st.cache_resource
def load_test_shrds():
    # opening the shards and slices
    with open('Data_and_Files/test_shards_npy.shrds', 'rb') as fh:
        test_shards = dill.load(fh)
        fh.close()
    return test_shards

@st.cache_resource
def load_train_shrds():
    with open('Data_and_Files/train_shards_npy.shrds', 'rb') as fh:
        train_shards = dill.load(fh)
        fh.close()
    return train_shards

@st.cache_resource
def load_test_sls():
    with open('Data_and_Files/test_slices_npy.sls', 'rb') as fh:
        test_slices = dill.load(fh)
        fh.close()
    return test_slices

@st.cache_resource
def load_train_sls():
    with open('Data_and_Files/train_slices_npy.sls', 'rb') as fh:
        train_slices = dill.load(fh)
    return train_slices

@st.cache_data
def load_df():
    df = pd.read_csv('Data_and_Files/fnma.loan.data.csv')
    features_list = [
    'Record Number',
    "Borrower’s (or Borrowers’) Annual Income",
    "Area Median Family Income (2016)",
    "Tract Income Ratio",
    "2010 Census Tract - Median Income",
    "Purpose of Loan",
    "Number of Borrowers",
    "First-Time Home Buyer",
    "Federal Guarantee",
    "Property Type",
    "Rate Spread",
    "Occupancy Code",
    "US Postal State Code",
    "Metropolitan Statistical Area (MSA) Code",
    "Lien Status"
    ]
    df2 = df[features_list]
    return df2

@st.cache_resource
def load_scaler():
    with open('Data_and_Files/scaler.scl', 'rb') as fh:
        scaler = dill.load(fh)
        fh.close()
    return scaler

# function finding out the exact location of the recod to be deleted. 
def find_shard_slice(to_delete: int, df: pd.DataFrame, scaler: StandardScaler, num_shards: int, num_slices: int, training_slices: list, testing_slices: list, X_train: torch.tensor, X_test: torch.tensor) -> tuple:
    '''Finds the shard index, slice index and index of the specific data point in the slice and returns them as a tuple'''
    to_delete = df[df['Record Number'] == to_delete]
    if to_delete.empty:
        return None
    to_delete = torch.tensor(scaler.transform(to_delete.values)).to(torch.float32)
    # for training dataset
    for shard_idx in range(num_shards):
        for slice_idx in range(num_slices):
            for idx in range(len(training_slices[shard_idx][slice_idx].x_slice)):
                if torch.allclose(to_delete, torch.tensor(training_slices[shard_idx][slice_idx].x_slice[idx])):
                   for train_idx in range(len(X_train)):
                       if torch.allclose(to_delete, X_train[train_idx]):
                           return (0, shard_idx, slice_idx, idx, train_idx) # 0 is for training dataset
    
    # for testing data set
    for shard_idx in range(num_shards):
        for slice_idx in range(num_slices):
            for idx in range(len(testing_slices[shard_idx][slice_idx].x_slice)):
                if torch.allclose(to_delete, torch.tensor(testing_slices[shard_idx][slice_idx].x_slice[idx])):
                   st.write('Record was found in the Testing Dataset')
                   for test_idx, test_pt in enumerate(X_test):
                       if torch.allclose(to_delete, test_pt):
                           return (1, shard_idx, slice_idx, idx, test_idx)
    return None
# function to delete the data point               
def del_data(record_no: int, test_or_train: int, shard_idx: int, slice_idx: int, idx: int, train_test_idx: int, test_slices: list, train_slices: list, df: pd.DataFrame, X_test: torch.tensor, X_train: torch.tensor, y_test: torch.tensor, y_train: torch.tensor):
    '''Takes the shard index, slice index and the index of the data point and deletes it from either the training or the testing data'''
    df.drop(df[df['Record Number'] == record_no].index, inplace=True)  # deleting it from the dataframe
    if test_or_train == 0:
        new_x_slice = np.concatenate((train_slices[shard_idx][slice_idx].x_slice[:idx], train_slices[shard_idx][slice_idx].x_slice[idx+1:]))
        new_y_slice = np.concatenate((train_slices[shard_idx][slice_idx].y_slice[:idx], train_slices[shard_idx][slice_idx].y_slice[idx+1:]))
        train_slices[shard_idx][slice_idx].x_slice = new_x_slice
        train_slices[shard_idx][slice_idx].y_slice = new_y_slice
        X_train = torch.cat([X_train[:train_test_idx], X_train[train_test_idx + 1:]])
        y_train = torch.cat([y_train[:train_test_idx], y_train[train_test_idx + 1:]])
        st.write(f"Data of training shard {shard_idx} in slice {slice_idx} at index of {idx}; Data in main dataset at index {train_test_idx} deleted successfully")

    elif test_or_train == 1:
        new_x_slice = np.concatenate((test_slices[shard_idx][slice_idx].x_slice[:idx], test_slices[shard_idx][slice_idx].x_slice[idx+1:]))
        new_y_slice = np.concatenate((test_slices[shard_idx][slice_idx].y_slice[:idx], test_slices[shard_idx][slice_idx].y_slice[idx+1:]))
        test_slices[shard_idx][slice_idx].x_slice = new_x_slice
        test_slices[shard_idx][slice_idx].y_slice = new_y_slice
        X_test = torch.cat([X_test[:train_test_idx], X_test[train_test_idx + 1:]])
        y_test = torch.cat([y_test[:train_test_idx], y_test[train_test_idx + 1:]])
        st.write(f"Data of testing shard {shard_idx} in slice {slice_idx} at index of {idx}; Data in main dataset at index {train_test_idx} deleted successfully")
    

# function to unlearn the data point
def unlearn(test_or_train: int, num_slices: int, shard_idx_todel: int, slice_idx_todel: int, train_slices, shard_models, BATCH_SIZE: int, num_epochs: int):
    if test_or_train == 0:
        model = shard_models[shard_idx_todel].model
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        total_steps = (num_slices + 1) - (slice_idx_todel + 1)
        progress_bar = st.progress(0)
        status_text = st.empty()
        steps = 0
        if slice_idx_todel == 0:
            shard_models[shard_idx_todel].model_params = []
            unl_model_params = []
            for slice_idx in range(num_slices):
                if slice_idx == 0:
                    x = torch.tensor(train_slices[shard_idx_todel][0].x_slice)
                    y = torch.tensor(train_slices[shard_idx_todel][0].y_slice)
                else:
                    x = torch.cat((x, torch.tensor(train_slices[shard_idx_todel][slice_idx].x_slice)))
                    y = torch.cat((y, torch.tensor(train_slices[shard_idx_todel][slice_idx].y_slice)))
                train_dataset = torch.utils.data.TensorDataset(x, y)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

                # training the model
                for epoch in range(num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        predictions = model(batch_X)
                        loss = criterion(predictions, batch_y)
                          
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                unl_model_params.append(model.state_dict())
                print('Shard:- {0}, Slice:- {1}'.format(shard_idx_todel, slice_idx))
                steps += 1
                progress_bar.progress((steps) / total_steps)
            shard_models[shard_idx_todel].model_params = unl_model_params
            status_text.text("Unlearning complete!")

        else:
            shard_models[shard_idx_todel].model_params = shard_models[shard_idx_todel].model_params[: slice_idx_todel - 1]
            unl_model_params = []
            model.load_state_dict(shard_models[shard_idx_todel].model_params[-1])

            for slice_idx in range(slice_idx_todel, num_slices):
                if slice_idx == slice_idx_todel:
                    x = torch.tensor(train_slices[shard_idx_todel][slice_idx].x_slice[:slice_idx + 1])
                    y = torch.tensor(train_slices[shard_idx_todel][slice_idx].y_slice[:slice_idx + 1])
                else:
                    x = torch.cat([x, torch.tensor(train_slices[shard_idx_todel][slice_idx].x_slice)])
                    y = torch.cat([y, torch.tensor(train_slices[shard_idx_todel][slice_idx].y_slice)])
                train_dataset = torch.utils.data.TensorDataset(x, y)
                train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

                for epoch in range(num_epochs):
                    model.train()
                    for batch_X, batch_y in train_loader:
                        predictions = model(batch_X)
                        loss = criterion(predictions, batch_y)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                unl_model_params.append(model.state_dict())
                print('Shard:- {0}, Slice:- {1}'.format(shard_idx_todel, slice_idx))
                steps += 1
                progress_bar.progress((steps) / total_steps)
            shard_models[shard_idx_todel].model_params[slice_idx_todel:] = unl_model_params
            status_text.text('Unlearning Complete!')


    else:
        st.write('The data was found in the testing shard and has no influence on the model.')


def calc_rmse(shard_models, X_train, y_train, X_test, y_test):
    '''Returns the tuple (train_rmse, test_rmse)'''
    sharded_test_predictions = []
    sharded_train_predictions = []
    for shard_model in shard_models:
        shard_model.model.eval()
        test_pred = (shard_model.model(X_test)).detach().numpy()
        train_pred = (shard_model.model(X_train)).detach().numpy()
        sharded_test_predictions.append(test_pred)
        sharded_train_predictions.append(train_pred)
    
    sharded_test_predictions = np.array(sharded_test_predictions)
    sharded_train_predictions = np.array(sharded_train_predictions)
    mean_test_predictions = np.mean(sharded_test_predictions, axis=0)
    mean_train_predictions = np.mean(sharded_train_predictions, axis=0)

    test_rmse = root_mean_squared_error(mean_test_predictions, y_test)
    train_rmse = root_mean_squared_error(mean_train_predictions, y_train)

    return (train_rmse, test_rmse)


if __name__ == '__main__':
    main()