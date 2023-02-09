import os
import json
import ants
import numpy as np
from tqdm import trange, tqdm

from runtime.utils import get_files_df


def compare_headers(header1, header2):
    is_valid = True

    if header1['dimensions'] != header2['dimensions']:
        is_valid = False
    if header1['spacing'] != header2['spacing']:
        is_valid = False

    return is_valid


class Analyze:

    def __init__(self, args):

        self.args = args
        with open(self.args.data, 'r') as file:
            self.data = json.load(file)

        self.labels = self.data['labels']
        self.train_dir = os.path.abspath(self.data['train-data'])
        self.output_dir = os.path.abspath(self.args.results)

        # Get paths to dataset
        if args.paths is None:
            self.train_paths_csv = os.path.join(self.output_dir, 'train_paths.csv')
        else:
            self.train_paths_csv = self.args.paths

        if self.args.config is None:
            self.config_file = os.path.join(self.args.results, 'config.json')
        else:
            self.config_file = self.args.config

        self.config = dict()
        self.df = get_files_df(self.data, 'train')

    def check_nz_mask(self):
        """
        If, on average, the image volume decreases by 25% after cropping zeros,
        use non-zero mask for the rest of the preprocessing pipeline.

        This reduces the size of the images and the memory foot print of the
        data i/o pipeline. An example of where this would be useful is the
        BraTS dataset.
        """

        print('Checking for non-zero mask...')

        image_vol_reduction = list()
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            image_list = list(patient.values())[2:len(patient)]

            # Read original image
            full_sized_image = ants.image_read(image_list[0])
            full_dims = full_sized_image.numpy().shape

            # Create non-zero mask from first image in image list
            nzmask = ants.image_read(image_list[0])
            nzmask = ants.get_mask(nzmask, cleanup=0)

            # Crop original image according to non-zero mask
            cropped_image = ants.crop_image(full_sized_image, nzmask)
            cropped_dims = cropped_image.numpy().shape

            image_vol_reduction.append(1. - (np.prod(cropped_dims) / np.prod(full_dims)))

        mean_vol_reduction = np.mean(image_vol_reduction)
        if np.mean(image_vol_reduction) >= 0.25:
            use_nz_mask = True
        else:
            use_nz_mask = False

        return use_nz_mask

    def get_target_spacing(self):
        """
        For non-uniform spacing, get median image spacing in each direction.
        This is median image spacing is our target image spacing for preprocessing.
        If data is isotropic, then set target spacing to given spacing in data.
        """

        print('Getting target spacing...')
        # If data is anisotrpic, get median image spacing
        original_spacings = np.zeros((len(self.df), 3))

        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()

            # Read mask image. This is faster to load.
            spacing = ants.image_header_info(patient['mask'])['spacing']

            # Get spacing
            original_spacings[i, :] = spacing

        # Initialize target spacing
        target_spacing = list(np.median(original_spacings, axis=0))
        is_anisotropic = False

        # Get the smallest and largest spacings
        spacing_min = np.min(np.min(original_spacings, axis=0))
        spacing_max = np.max(np.max(original_spacings, axis=0))

        if spacing_max / spacing_min > 3.:
            is_anisotropic = True
            largest_axis = list(np.unique(np.where(original_spacings == spacing_max)[-1]))
            trailing_axes = list({0, 1, 2} - set(largest_axis))

            if len(largest_axis) == 1:
                target_spacing[largest_axis[0]] = np.percentile(original_spacings[:, largest_axis[0]], 10)

                for ax in trailing_axes:
                    target_spacing[ax] = np.median(original_spacings[:, ax])

        return target_spacing, is_anisotropic

    def check_resampled_dims(self):
        """
        Determine dims from resampled data.
        """

        print('Checking resampled dimensions...')

        resampled_dims = np.zeros((len(self.df), 3))
        max_buffer_size = 1.99e9
        cnt = 0
        pbar = tqdm(total=len(self.df))

        while cnt < len(self.df):
            patient = self.df.iloc[cnt].to_dict()
            mask = ants.image_read(patient['mask'])
            image_list = list(patient.values())[2:len(patient)]

            if self.config['use_nz_mask']:
                # Create non-zero mask from first image in image list
                nzmask = ants.image_read(image_list[0])
                nzmask = ants.get_mask(nzmask, cleanup=0)

                # Crop mask and all images according to brainmask and pad with patch radius
                mask = ants.crop_image(mask, nzmask)

            else:
                # Use the mask for this. It is faster to load and resample.
                mask = ants.image_read(patient['mask'])

            dims = mask.numpy().shape

            # If native image spacing is different from target, get resampled image size
            if not(np.array_equal(np.array(mask.spacing), np.array(self.config['target_spacing']))):
                dims = [int(np.round((dims[i]*mask.spacing[i]) / self.config['target_spacing'][i]))
                        for i in range(len(dims))]

            # Get image buffer sizes
            image_buffer_size = 4 * (np.prod(dims) * (len(image_list) + len(self.labels)))

            # If data exceeds tfrecord buffer size, then resample to coarser resolution
            if image_buffer_size > max_buffer_size:
                print('Images are too large, coarsening target spacing...')

                if self.is_anisotropic:
                    trailing_dims = np.where(
                        self.config['target_spacing'] != np.max(self.config['target_spacing']))
                    for i in list(trailing_dims[0]):
                        self.config['target_spacing'][i] *= 2

                else:
                    for i in range(3):
                        self.config['target_spacing'][i] *= 2

                cnt = 0
                pbar.refresh()
                pbar.reset()
            else:
                resampled_dims[cnt, :] = dims
                cnt += 1
                pbar.update(1)

        pbar.close()

        # Get patch size after finalizing target image spacing
        median_resampled_dims = list(np.median(resampled_dims, axis=0))
        median_resampled_dims = [int(np.floor(median_resampled_dims[i])) for i in range(3)]

        return median_resampled_dims

    def get_ct_norm_parameters(self):
        """
        Get normalization parameters (i.e., window ranges and z-score) for CT images.
        """

        print('Getting CT normalization parameters...')

        fg_intensities = list()

        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            image_list = list(patient.values())[2:len(patient)]

            # Read original image
            image = ants.image_read(image_list[0]).numpy()

            # Get foreground mask and binarize it
            mask = ants.image_read(patient['mask'])
            mask = ants.get_mask(mask, cleanup=0).numpy()

            # Get foreground voxels in original image
            image = np.multiply(image, mask)

            # You don't need to use all of the voxels for this
            fg_intensities += list(image[image != 0][::10])

        global_z_score_mean = np.mean(fg_intensities)
        global_z_score_std = np.std(fg_intensities)
        global_window_range = [np.percentile(fg_intensities, 0.5),
                               np.percentile(fg_intensities, 99.5)]

        return global_z_score_mean, global_z_score_std, global_window_range

    def analyze_dataset(self):
        """
        Analyze dataset to get inferred parameters.
        """
        # Start getting parameters from dataset
        use_nz_mask = self.check_nz_mask()
        target_spacing, self.is_anisotropic = self.get_target_spacing()

        if self.data['modality'] == 'ct':
            # Get CT normalization parameters
            global_z_score_mean, global_z_score_std, global_window_range = self.get_ct_norm_parameters()

            self.config = {'modality': 'ct',
                           'labels': self.labels,
                           'use_nz_mask': use_nz_mask,
                           'target_spacing': [float(target_spacing[i]) for i in range(3)],
                           'window_range': [float(global_window_range[i]) for i in range(2)],
                           'global_z_score_mean': float(global_z_score_mean),
                           'global_z_score_std': float(global_z_score_std)}
        else:
            self.config = {'modality': self.data['modality'],
                           'labels': self.labels,
                           'use_nz_mask': use_nz_mask,
                           'target_spacing': [float(target_spacing[i]) for i in range(3)]}
        median_dims = self.check_resampled_dims()
        self.config['median_image_size'] = [int(median_dims[i]) for i in range(3)]

    def run(self):

        print('Analyzing dataset...')

        '''
        Check if headers match up. Remove data whose headers to not match.
        '''
        print('Verifying dataset integrity...')
        bad_data = list()
        for i in trange(len(self.df)):
            patient = self.df.iloc[i].to_dict()
            mask_header = ants.image_header_info(patient['mask'])

            # Get list of image paths and segmentation mask
            image_list = list(patient.values())[2:len(patient)]
            for image_path in image_list:
                image_header = ants.image_header_info(image_path)

                is_valid = compare_headers(mask_header, image_header)

                if not is_valid:
                    print('In {}: Mismatch between image and mask header information'.format(patient['id']))
                    bad_data.append(i)
                    break

            if len(image_list) > 1:
                anchor_image = image_list[0]
                anchor_header = ants.image_header_info(anchor_image)

                for image_path in image_list[1:]:
                    image_header = ants.image_header_info(image_path)

                    is_valid = compare_headers(anchor_header, image_header)

                    if not is_valid:
                        print('In {}: Mismatch between images header information'.format(patient['id']))
                        bad_data.append(i)
                        break

        rows_to_drop = self.df.index[bad_data]
        self.df.drop(rows_to_drop, inplace=True)
        self.df.to_csv(self.train_paths_csv, index=False)

        # Get inferred parameters
        self.analyze_dataset()

        # Save inferred parameters as json file
        with open(self.config_file, 'w') as outfile:
            json.dump(self.config, outfile, indent=2)
