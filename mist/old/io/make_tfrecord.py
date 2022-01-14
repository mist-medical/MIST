import pandas as pd
import tensorflow as tf

def make_tfrecords(df, filename):
    '''
    Make a tfrecord version of nifti dataset.

    Input:
    df:     dataframe with filepaths to preprocessed data
            columns - id, mask, image 1, ..., image n
    filename: name of tfrecord file to be created
    '''
    def _float_feature(value):
        return tf.train.Feature(float_list = tf.train.FloatList(value = value))
    def _int_feature(value):
        return tf.train.Feature(int64_list = tf.train.Int64List(value = value))

    # open the file
    writer = tf.io.TFRecordWriter(filename,
                                  options = tf.io.TFRecordOptions(compression_type = 'GZIP'))

    for i in trange(len(df)):
        patient = df.iloc[i].to_dict()
        mask_info = ants.image_header_info(patient['mask'])
        dims = mask_info['dimensions']
        dims = tuple(int(d) for d in dims)

        # Generalize this...
        # Mask labels and patch_size should be available everywhere
        mask_labels = [0, 1, 2, 4]
        patch_size = 64
        radius = patch_size // 2

        # Clean up variable names here...
        mask_numpy = ants.image_read(patient['mask']).numpy()
        mask_numpy = np.pad(mask_numpy, radius)
        mask = np.empty((*(dim + patch_size for dim in dims), len(mask_labels)))
        for j in range(len(mask_labels)):
            mask[..., j] = mask_numpy == mask_labels[j]

        # Clean up variable names here...
        image_list = list(patient.values())[2:len(patient)]
        image = np.empty((*(dim + patch_size for dim in dims), len(image_list)))
        for j in range(len(image_list)):
            image_ants = ants.image_read(image_list[j])
            brainmask = ants.get_mask(image_ants, cleanup = 0).numpy()
            image_numpy = image_ants.numpy()
            image_numpy = normalize(image_numpy, brainmask)

            # Pad each image with radius to ensure each point in mask can be picked
            image[..., j] = np.pad(image_numpy, radius)

        # Get indicies of foreground values in mask
        fg_mask = (mask[..., 0] == 0).astype('int')
        nonzero_fg = np.nonzero(fg_mask)
        nonzero_fg = np.vstack((nonzero_fg[0], nonzero_fg[1], nonzero_fg[2]))
        num_nonzero_fg = [nonzero_fg.shape[-1]]

        # Get indicies of background values in mask
        # Background needs to be brain mask - tumor mask
        # Need to add options to make this more generalizable
        # For neuro imaging, generally have brainmask, but that is not the case for CT and general MR
        bg_mask = (image[..., 0] > 0).astype('int') - fg_mask
        nonzero_bg = np.nonzero(bg_mask)
        nonzero_bg = np.vstack((nonzero_bg[0], nonzero_bg[1], nonzero_bg[2]))
        num_nonzero_bg = [nonzero_bg.shape[-1]]

        # Create a feature
        feature = {'image': _float_feature(image.ravel()),
                   'mask': _float_feature(mask.ravel()),
                   'fg_points': _int_feature(nonzero_fg.ravel()),
                   'num_fg_points': _int_feature(num_nonzero_fg),
                   'bg_points': _int_feature(nonzero_bg.ravel()),
                   'num_bg_points': _int_feature(num_nonzero_bg)}

        # Create an example protocol buffer
        example = tf.train.Example(features = tf.train.Features(feature = feature))

        # Serialize to string and write on the file
        writer.write(example.SerializeToString())

    writer.close()
