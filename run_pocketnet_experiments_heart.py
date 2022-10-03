import subprocess

processed_data = "/workspace/data/pocketnet/task02_heart/numpy"
data_json = "/workspace/data/msd/mist/Task02_Heart/LeftAtrium_dataset.json"
config = "/workspace/data/pocketnet/task02_heart/pocket_nnunet/config.json"
paths = "/workspace/data/pocketnet/task02_heart/pocket_nnunet/train_paths.csv"
architectures = ["nnunet", "unet", "resnet", "densenet", "hrnet"]
states = ["pocket", "full"]
gpu_num = 4
cnt = 0

for arch in architectures:
    for state in states:
        results = "/workspace/data/pocketnet/task02_heart/{}_{}".format(state, arch)

        if cnt == 0:
            mode = "train"
        else:
            mode = "train"

        cmd = "python main.py "
        cmd += "--exec-mode {} ".format(mode)
        cmd += "--data {} ".format(data_json)
        cmd += "--processed-data {} ".format(processed_data)
        cmd += "--config {} ".format(config)
        cmd += "--paths {} ".format(paths)
        cmd += "--results {} ".format(results)
        cmd += "--model {} ".format(arch)
        cmd += "--gpus {} ".format(gpu_num)
        cmd += "--epochs 300 "
        cmd += "--loss dice_ce "
        cmd += "--xla "
        cmd += "--amp "
        cmd += "--patch-size 128 128 128 "
        cmd += "--seed 42 "
        cmd += "--steps-per-epoch 50 "

        if state == "pocket":
            cmd += " --pocket"

        subprocess.call(cmd, shell=True)
        cnt += 1
