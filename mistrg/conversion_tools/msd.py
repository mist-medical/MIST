import os
import json
import pprint
import subprocess
import numpy as np
import SimpleITK as sitk
from tqdm import trange


def create_empty_dir(path):
    if not (os.path.exists(path)):
        os.mkdir(path)


def copy_msd_data(source, dest, msd_json, modalities, mode):
    # Copy data to destination in MIST format
    for i in trange(len(msd_json[mode])):
        if mode == 'training':
            pat = os.path.basename(msd_json[mode][i]['image']).split('.')[0]
            image = os.path.join(source, 'imagesTr', '{}.nii.gz'.format(pat))
            mask = os.path.join(source, 'labelsTr', '{}.nii.gz'.format(pat))
            patient_dir = os.path.join(dest, 'raw', 'train', pat)
        if mode == 'test':
            pat = os.path.basename(msd_json[mode][i]).split('.')[0]
            image = os.path.join(source, 'imagesTs', '{}.nii.gz'.format(pat))
            patient_dir = os.path.join(dest, 'raw', 'test', pat)

        create_empty_dir(patient_dir)

        # If images are 4D, split them into individal 3D images
        if len(modalities) > 1:
            # Read image as sitk image. ANTs is not good for this bit...
            image_sitk = sitk.ReadImage(image)
            image_npy = sitk.GetArrayFromImage(image_sitk)

            # Get individual direction
            direction = np.array(image_sitk.GetDirection()).reshape((4, 4))
            direction = direction[0:3, 0:3]
            direction = np.ravel(direction)
            direction = tuple(direction)

            # Get individual spacing
            spacing = image_sitk.GetSpacing()
            spacing = spacing[:-1]

            # Get individual origin
            origin = image_sitk.GetOrigin()
            origin = origin[:-1]

            for j in range(image_npy.shape[0]):
                # Get image array
                img_j = image_npy[j, ...]

                # Convert to sitk image
                img_j = sitk.GetImageFromArray(img_j)

                # Set direction, spacing, and origin
                img_j.SetDirection(direction)
                img_j.SetSpacing(spacing)
                img_j.SetOrigin(origin)

                # Write individual image to nifit
                output_name = os.path.join(patient_dir, '{}.nii.gz'.format(modalities[j]))
                sitk.WriteImage(img_j, output_name)
        else:
            copy_cmd = 'cp {} {}'.format(image, os.path.join(patient_dir, '{}.nii.gz'.format(modalities[0])))
            subprocess.call(copy_cmd, shell=True)

        if mode == 'training':
            # Copy mask to destination
            copy_cmd = 'cp {} {}'.format(mask, os.path.join(patient_dir, 'mask.nii.gz'))
            subprocess.call(copy_cmd, shell=True)


# Convert MSD data to MIST format
def convert_msd(source, dest):
    source = os.path.abspath(source)
    dest = os.path.abspath(dest)

    if not (os.path.exists(source)):
        raise Exception('{} does not exist!'.format(source))

    # Create destination folder and sub-folders
    create_empty_dir(dest)
    create_empty_dir(os.path.join(dest, 'raw'))
    create_empty_dir(os.path.join(dest, 'raw', 'train'))

    exists_test = False
    if os.path.exists(os.path.join(source, 'imagesTs')):
        exists_test = True
        create_empty_dir(os.path.join(dest, 'raw', 'test'))

    msd_json_path = os.path.join(source, 'dataset.json')
    if not (os.path.exists(msd_json_path)):
        raise Exception('{} does not exist!'.format(msd_json_path))

    with open(msd_json_path, 'r') as file:
        msd_json = json.load(file)

    # Get modalities
    modalities = dict()
    for idx in msd_json['modality'].keys():
        modalities[int(idx)] = msd_json['modality'][idx]

    # Copy data to destination in MIST format
    print('Converting training data to MIST format...')
    copy_msd_data(source, dest, msd_json, modalities, 'training')

    if exists_test:
        print('Converting test data to MIST format...')
        copy_msd_data(source, dest, msd_json, modalities, 'test')

    # Create MIST dataset json file
    dataset_json = dict()

    # Get task name
    dataset_json['task'] = msd_json['name']

    # Handel modalities input
    modalities_list = list(modalities.values())
    modalities_list = [m.lower() for m in modalities_list]
    if 'ct' in modalities_list:
        dataset_json['modality'] = 'ct'
    elif 'mri' in modalities_list:
        dataset_json['modality'] = 'mr'
    else:
        dataset_json['modality'] = 'other'

    # Get training and testing directories
    dataset_json['train-data'] = os.path.abspath(os.path.join(dest, 'raw', 'train'))
    if exists_test:
        dataset_json['test-data'] = os.path.abspath(os.path.join(dest, 'raw', 'test'))

    # Handel mask/images input
    dataset_json['mask'] = ['mask.nii.gz']
    images_dict = dict()
    for i in range(len(modalities)):
        images_dict[modalities[i]] = ['{}.nii.gz'.format(modalities[i])]
    dataset_json['images'] = images_dict

    # Handel labels and classes input
    labels = dict()
    for idx in msd_json['labels'].keys():
        labels[int(idx)] = msd_json['labels'][idx]
    dataset_json['labels'] = list(labels.keys())

    final_classes_dict = dict()
    for label in labels.keys():
        if label != 0:
            final_classes_dict[labels[label].replace(' ', '_')] = [label]
    dataset_json['final_classes'] = final_classes_dict

    # Write MIST formated dataset to json file
    dataset_json_filename = os.path.join(dest, '{}_dataset.json'.format(dataset_json['task']))
    print('MIST dataset parameters written to {}\n'.format(dataset_json_filename))
    pprint.pprint(dataset_json, sort_dicts=False)
    print('')

    with open(dataset_json_filename, 'w') as outfile:
        json.dump(dataset_json, outfile, indent=2)
