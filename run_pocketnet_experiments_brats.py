import subprocess

processed_data = "/workspace/data/pocketnet/brats/numpy"
data_json = "/workspace/github/mist_memory_test/mist_dali_loader/brats2020.json"
config = "/workspace/data/pocketnet/brats/pocket_nnunet/config.json"
paths = "/workspace/data/pocketnet/brats/pocket_nnunet/train_paths.csv"
architectures = ["nnunet", "unet", "resnet", "densenet", "hrnet"]
states = ["pocket", "full"]
gpu_num = 6
cnt = 0

for arch in architectures:
    for state in states:
        results = "/workspace/data/pocketnet/brats/{}_{}".format(state, arch)

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

        if state == "pocket":
            cmd += " --pocket"

        subprocess.call(cmd, shell=True)
        cnt += 1
