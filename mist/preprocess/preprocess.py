

# Create a CSV with file paths for a dataset
def get_paths_csv(base_dir, name_dict, output_csv):
    def get_files(path):
        files_list = list()
        for root, _, files in os.walk(path, topdown = False):
            for name in files:
                files_list.append(os.path.join(root, name))
        return files_list

    cols = ['id'] + list(name_dict.keys())
    df = pd.DataFrame(columns = cols)
    row_dict = dict.fromkeys(cols)

    ids = os.listdir(base_dir)

    for i in ids:
        row_dict['id'] = i
        path = os.path.join(base_dir, i)
        files = get_files(path)

        for file in files:
            for img_type in name_dict.keys():
                for img_string in name_dict[img_type]:
                    if img_string in file:
                        row_dict[img_type] = file

        df = df.append(row_dict, ignore_index = True)

    df.to_csv(output_csv, index = False)

    ################# End of function #################
