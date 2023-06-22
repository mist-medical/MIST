import os
import json
import pprint
import subprocess
import pandas as pd
from tqdm import trange

def create_empty_dir(path):
    if not(os.path.exists(path)):
        os.mkdir(path)   

def copy_csv_data(df, dest, mode):
    for i in trange(len(df)):
        patient = df.iloc[i].to_dict()

        # Create new patient folder
        patient_dest = os.path.join(dest, str(patient['id']))
        create_empty_dir(patient_dest)
        
        if mode == 'train':
            # Copy mask to new patient folder
            mask_source = os.path.abspath(patient['mask'])
            mask_dest = os.path.join(patient_dest, 'mask.nii.gz')
            cp_mask_cmd = 'cp {} {}'.format(mask_source, mask_dest)
            subprocess.call(cp_mask_cmd, shell = True)

        # Copy images to new patient folder
        if mode == 'train':
            image_keys = list(patient.keys())[2:len(patient)]
            image_list = list(patient.values())[2:len(patient)]
        else:
            image_keys = list(patient.keys())[1:len(patient)]
            image_list = list(patient.values())[1:len(patient)]
            
        for j in range(len(image_keys)):
            image_source = os.path.abspath(image_list[j])
            image_dest = os.path.join(patient_dest, '{}.nii.gz'.format(image_keys[j]))
            cp_image_cmd = 'cp {} {}'.format(image_source, image_dest)
            subprocess.call(cp_image_cmd, shell = True)
    
def convert_csv(train_csv, test_csv, dest):
    dest = os.path.abspath(dest)
    
    # Check if inputs exist
    if not(os.path.exists(train_csv)):
        raise Exception('{} does not exist!'.format(train_csv))
    else:
        train_csv = os.path.abspath(train_csv)
        
    exists_test = False
    if not(test_csv is None):
        exists_test = True
        if not(os.path.exists(test_csv)):
            raise Exception('{} does not exist!'.format(test_csv))
        else:
            train_csv = os.path.abspath(train_csv)
            
    # Create destination directories
    create_empty_dir(dest)
    
    train_dest = os.path.join(dest, 'raw', 'train')
    create_empty_dir(os.path.join(dest, 'raw'))
    create_empty_dir(train_dest)
    
    if exists_test:
        test_dest = os.path.join(dest, 'raw', 'test')
        create_empty_dir(test_dest)
        
    print('Converting training data to MIST format...')
    train_df = pd.read_csv(train_csv)
    copy_csv_data(train_df, train_dest, 'train')
    
    if exists_test:
        print('Converting test data to MIST format...')
        test_df = pd.read_csv(test_csv)
        copy_csv_data(test_df, test_dest, 'test')
        
    # Create MIST dataset json file
    dataset_json = dict()
    
    # Get task name
    dataset_json['task'] = None
    
    # Handel modalities input
    dataset_json['modality'] = None

    # Get training and testing directories
    dataset_json['train-data'] = os.path.abspath(train_dest)
    if exists_test:
        dataset_json['test-data'] = os.path.abspath(test_dest)
        
    # Handel mask/images input
    dataset_json['mask'] = ['mask.nii.gz']
    
    images_dict = dict()
    modalities = list(train_df.columns)[2:]
    for i in range(len(modalities)):
        images_dict[modalities[i]] = ['{}.nii.gz'.format(modalities[i])]
    dataset_json['images'] = images_dict
    
    # Handel labels and classes input
    dataset_json['labels'] = None
    dataset_json['final_classes'] = None
    
    # Write MIST formated dataset to json file
    dataset_json_filename = os.path.join(dest, 'dataset.json')
    print('MIST dataset parameters written to {}\n'.format(dataset_json_filename))
    pprint.pprint(dataset_json, sort_dicts = False)
    print('')
    print('Please add task, modality, labels, and final classes to parameters')
    
    with open(dataset_json_filename, 'w') as outfile:
        json.dump(dataset_json, outfile, indent = 2)

    
            
    