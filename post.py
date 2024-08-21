"""
Main program

NOTE: The "run" in running mode here means we do both train and infer

@yuningw
"""

case_numbers = range(1, 1+1)

import torch
from lib import train_UNet, train_LSTM, test_UNet, test_LSTM, preprocessing, plot, hyperparameter, get_mode, visualize_latent, parser
from models import network
import h5py



if __name__ == "__main__":


    # Print data info
    # datafile = "./data/Deepmind_valid.h5"
    # data = h5py.File(datafile, 'r')
    # trainlist_ = [str(i) for i in range(100)]
    # trainlist = ",".join(trainlist_)
    # train_data_list, trainlist = preprocessing.get_data_from_idx(data, trainlist)
    # print("#"*10 + "  Train dataset info  " + "#"*10)
    # for enu, single_data in enumerate(train_data_list):
	#     print(f'idx {enu}')
	#     for item in single_data.keys():
	#         print('{} : {}'.format(item, single_data[item].shape))

	# Post-process summary csv files


    import os
    import pandas as pd

    # Path to the directory containing your CSV files
    directory = './' + 'case_folder' + '/'
    # file_pattern = 'C{}_Err_summary.csv'


    # Initialize an empty DataFrame to accumulate all data
    accumulated_data = pd.DataFrame()
    accumulated_mean_data = []
    accumulated_std_data = []
    # Process each file and append the required rows
    for case_number in case_numbers:
        file_path = os.path.join(directory+f'C{case_number}/', f'[Avg]C{case_number}_Err_summary.csv')
        data = pd.read_csv(file_path)  # Read only the first two rows
        mean_data = data.iloc[0:1].values.flatten().tolist()  # 2nd row for mean error
        std_data = data.iloc[1:2].values.flatten().tolist()  # 3rd row for STD error
        case_label = f'C{case_number}'

        # Insert the case number at the beginning
        mean_data.insert(0, case_label)
        std_data.insert(0, case_label)

        # Append to the lists
        accumulated_mean_data.append(mean_data)
        accumulated_std_data.append(std_data)
        # data.insert(0, 'Case', f'C{case_number}')  # Insert the case number as the first column
        # accumulated_data = pd.concat([accumulated_data, data], ignore_index=True)

    # Output file path
    accumulated_data = accumulated_mean_data + accumulated_std_data

    # Create a DataFrame
    accumulated_df = pd.DataFrame(accumulated_data,
                                  columns=['Case', 'Error Type']+ data.columns.tolist()[1:])
                                  # columns=['Case', 'Error Type', 'Idx 4', 'Idx 6', 'Idx 8', 'Idx 10', 'Idx 15',
                                  #          'Idx 16'])
    # accumulated_data = pd.concat(accumulated_mean_data + accumulated_std_data)
    output_file_path = os.path.join(directory, f'C{case_numbers[0]}-{case_numbers[-1]}_Err_summary.csv')


    # Save the accumulated data to a new CSV file
    accumulated_df.to_csv(output_file_path, index=False)