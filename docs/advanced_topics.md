Advanced Topics
===

## Docker
The MIST package is also available as a Docker image. Start by pulling the ```mistmedical/mist``` image from DockerHub:
```console
docker pull mistmedical/mist:latest
```

Use the following command to run an interactive Docker container with the MIST package:
```console
docker run --rm -it -u $(id -u):$(id -g) --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -v /your/working/directory/:/workspace mist
```

From there, you can run any of the commands described above inside the Docker container. Additionally, you can use the Docker 
entrypoint command to run any of the MIST scripts.

## Multi-GPU Training
We use PyTorch's DistributedDataParallel (DDP) for multi-GPU data parallelism. By default, 
MIST will use all visible GPUs on a given system. However, you may specify which GPUs MIST uses for 
training with the ```--gpus``` flag for the ```mist_run_all``` and ```mist_train``` commands. 

For example, if your system has eight GPUs available, but you only want to 
use GPUs 0 and 5, then run ```mist_run_all <other arguments> --gpus 0 5```.

Note that the ```--master-port``` flag (default 12355), sets the port for 
multiple GPUs to communicate on the host device.

## Network Architectures
The default architecture for MIST is nnUNet. However, MIST offers a variety
of other architectures that can be used with the ```--model``` flag for 
the ```mist_run_all``` and ```mist_train``` commands. The table below 
lists the available architectures and the corresponding input for 
the ```--model``` flag.

| Architecture        | ```--model```                                                                                                                                                 |
|---------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| U-Net               | ```unet```                                                                                                                                                    |
| Attention U-Net     | ```attn_unet```                                                                                                                                               |
| Swin UNETR               | ```swin-unetr```                                                                                                                                                   |
| FMG-Net             | ```fmgnet```                                                                                                                                                  |
| W-Net               | ```wnet```                                                                                                                                                    |
| nnUNet              | ```nnunet```                                                                                                                                                  |

## Regularization
There are several regularization features available to each architecture which can be specified for the 
the ```mist_run_all``` and ```mist_train``` commands.

* ```--l2-reg```: (default: False) Turns on L2 regularization with the penalty term set by ```--l2-penalty``` flag, which has a default value of 1e-05.
This feature is available for all network architectures.
* ```--l1-reg```: (default: False) Turns on L1 regularization with the penalty term set by ```--l1-penalty``` flag, which has a default value of 1e-05.
This feature is available for all network architectures.
* ```--vae-reg```: (default: False) Turns on variational autoencoder (VAE) regularization for the nnUNet, U-Net, FMG-Net, and W-Net architectures 
with the penalty term set by ```--vae-penalty```, which has a default value of 0.01.
* ```--deep-supervision```: (default: False) Turns on deep supervision for the nnUNet, U-Net, FMG-Net, and W-Net architectures with the 
number of supervision heads set by ```--deep-supervision-heads```, which has a default value of 2.

## Transfer Learning
MIST supports transfer learning with previously trained MIST models. To use a pretrained MIST model as an initial set of 
weights for a new dataset, set the ```--model``` flag to ```pretrained``` and give the path to the pretrained MIST model
directory using the ```--pretrained-model-path``` flag. 

!!! note
    Do not point ```--pretrained-model-path``` to an individual model, but the entire ```models/``` folder. MIST will
    automatically average all of the individual model weights and replace the input and output layers to match the number
    input and output channels in your new data.

For example, to use transfer learning with ```mist_run_all``` or ```mist_train```, use something like the following
command

```console
mist_run_all --model pretrained --pretrained-model-path /path/to/other/mist/models/directory --<other arguments> 
```

## Optimizers and Learning Rate Schedules
During training, MIST uses the Adam optimizer with a constant learning rate of 0.0003 by default. However, other optimizers
and learning rate schedules are available for the ```mist_run_all``` and ```mist_train``` commands. The following flags/inputs
will change the optimizer, learning rate, learning rate schedule, or apply gradient clipping.

* ```--optimizer```: (default: ```adam```) Sets optimizer for training. Other options are ```sgd``` and ```adamw```
* ```--learning-rate```: (default: 0.0003) Initial learning rate for training
  * ```--lr-scheduler```: (default: ```constant```) Sets the learning rate scheduler for training. Other options are 
  ```cosine-warm-restarts```, ```cosine```, and ```polynomial```.
* ```--clip-norm```: (default: False) Use gradient clipping (global)
    - ```--clip-norm-max``` (default: 1)  Max threshold for global norm clipping
* ```--sgd-momentum```: (default: 0) Momentum for SGD optimizer

## Loss Functions
MIST provides several loss functions and supports boundary-based loss functions. By default, MIST uses the Dice with
Cross Entropy loss function. The following loss functions are available:

| Loss                          | ```--loss```  |
|-------------------------------|---------------|
| Dice w/ Cross Entropy         | ```dice_ce``` |
| Dice                          | ```dice```    |
| Hausdorff Loss                | ```hdl```     |
| Boundary Loss                 | ```bl```      |
| Generalized Surface Loss      | ```gsl```     |
| clDice                        | ```cldice```  |

!!! warning
    To use boundary-based loss functions like the ```hdl```, ```bl```, or ```gsl```, you must use the ```--use-dtms```
    flag with ```mist_run_all``` or ```mist_preprocess``` and ```mist_train```. This will tell the MIST pipeline to 
    compute the distance transform maps (DTMs) of the ground truth masks during preprocessing and use them during training. 

## Misc. Topics

### Weighting Schedules for Boundary-based Loss Functions
Boundary-based loss functions are often used with region-based loss functions (i.e., the Dice loss) in the following
weighted combination:

$$
\alpha \mathcal{L}_{region} + (1 - \alpha) \mathcal{L}_{boundary},
$$

where $\alpha$ is a weighting parameter that can be constant or vary depending on the current epoch. MIST implements
several weighting schemes for $\alpha$. These weighting schemes and their corresponding flags are given below

* ```--boundary-loss-schedule```: (default: ```constant```) Weighting schedule for boundary losses. Options are 
```constant```, ```linear```, ```step```, ```cosine```. These schedules are described below
    - ```constant```: Constant schedule for $\alpha$ that can be set with the ```--loss-schedule-constant``` (default: 0.5)
    - ```linear```: Schedule for $\alpha$ that starts with $\alpha=1$ and linearly decreases it to $\alpha=0$. You 
    can pause this schedule for a certain number of epochs with ```--linear-schedule-pause``` (default: 5)
    - ```cosine```: Schedule for $\alpha$ that starts with $\alpha=1$ and decreases via a cosine function to $\alpha=0$
    - ```step```: Schedule for $\alpha$ that starts with $\alpha=1$ and decreases via a step function to $\alpha=0$. The 
    length of each step (in epochs) can be set with ```--step-schedule-step-length``` (default: 5)

### Patch Size Selection
During the analysis phase of the MIST pipeline, we compute the median resampled image size. We set the patch size for 
training from this median image size by taking the nearest power of two less than or equal to each dimension
in the median image size up to a ```--max-patch-size```. The default value for ```--max-patch-size``` is ```[256 256 256]```,
which corresponds to a patch size of 256$\times$256$\times$256. Users who know their system better than we can update this maximum patch size.

The default value of the patch size is written in the ```config.json``` file after the analysis portion of the pipeline.
You can change the patch size by setting the ```--patch-size``` argument, which is a list of the patch dimensions in the
$x$, $y$, and $z$ directions.

For example, you can update the ```--max-patch-size``` and/or the ```--patch-size``` in ```mist_run_all``` or ```mist_train```
with something like the following command

```console
mist_run_all --max-patch-size 256 256 128 --patch-size 128 128 128 --<other arguments>
```

### Validation Split Size/Patch Overlap
MIST computes the validation loss every 250 optimization steps. By default, MIST
will use a five-fold cross-validation and use the 20% hold out for validation.
However, if you want to partition the training set into a train and validation
set and use the 20% hold out as an independent test set, then set the
```--val-percent``` argument to something in the interval (0, 1). If you are
dealing with a large dataset, validation can add a considerable amount of time
to training. To speed up validation, you can adjust the amount of overlap
between patches using the ```--val-sw-overlap``` (default: 0.5) to a smaller
amount (i.e., 0.25) to speed up validation. For test time inference, MIST
uses the value stored in the ```--sw-overlap``` flag to control the amount of
overlap between patches.

In addition to these features, users can also set validation to happen
periodically after a certain number of epochs with the ```--validate-every-n-epochs```
and ```--validate-after-n-epochs``` arguments. For example, if you want to
validate every five epochs after 100 epochs, then use something like the following
command

```console
mist_run_all --validate-every-n-epochs 5 --validate-after-n-epochs 100 --<other arguments>
```

### Kubernetes
For MD Anderson users, you can run MIST on the Kubernetes/HPC cluster. Here is an example of a job submission file:

```yaml
---
apiVersion: batch/v1
kind: Job
metadata:
  name: <your job name goes here>
  namespace: yn-gpu-workload
  labels:
      k8s-user: <get this from your k8s-templates folder>
spec:
  backoffLimit: 0
  ttlSecondsAfterFinished: 600
  template:
    spec:
      nodeSelector:
        "nvidia.com/gpu.present": "true"
        "gpu-type": "A100" # Change this "gpu-type": "H100" for H100s.
      securityContext:
        runAsUser: <get this from your k8s-templates folder>
        runAsGroup: <get this from your k8s-templates folder>
        fsGroup: <get this from your k8s-templates folder>
      containers:
        - name: main
          image: mistmedical/mist:latest # Check https://hub.docker.com/r/mistmedical/mist for latest version.
          command: ["/bin/bash", "-c"]
          args: ["mist_run_all 
          --data $HOME/path/to/your/dataset.json 
          --numpy $HOME/path/to/your/numpy 
          --results $HOME/path/to/your/results"]
          workingDir: <get this from your k8s-templates folder>
          env:
          - name: HOME
            value: <get this from your k8s-templates folder>
          volumeMounts:
            - name: shm
              mountPath: "/dev/shm"
            - name: home
              mountPath: <get this from your k8s-templates folder>
          resources:
            limits:
              nvidia.com/gpu: "1" # Change this to increase number of GPUs, max of 8.
          imagePullPolicy: Always
      volumes:
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: '21474836480'
        - name: home
          persistentVolumeClaim:
            claimName: <get this from your k8s-templates folder>
      restartPolicy: Never
```

Once you save your job submission file, you can run it with

```console
kubectl apply -f <your submission file here>
```