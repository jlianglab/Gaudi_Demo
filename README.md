# Using Gaudi HPUs for Medical Image Analysis with Deep Learning Models

This repository details how to train and test PyTorch models with Intel Gaudi2 HPUs, specifically for medical image analysis. Gaudis are designed for AI processing rather than general parallel processing like NVIDIA GPUs. Gaudi integrates with PyTorch using Habana Labs' SynapseAI software. More information here: [Official Documentation](https://docs.habana.ai/en/latest/PyTorch/Getting_Started_with_PyTorch_and_Gaudi/Getting_Started_with_PyTorch.html)

## Installation

A container ([Apptainer](https://docs.rc.asu.edu/apptainer)) is used to ensure reproducibility and package persistence. The container pulls the base image from Habana's custom Docker image, which contains a modified version of PyTorch. When you run the container, it mounts the working directory as ```/workspace/```. This repository provides two containers, one that supports NVIDIA GPUs and one that supports Gaudi HPUs. The code is compatible with either processing unit.

1. Clone the GitHub repository:

```bash
git clone https://github.com/echang818/Gaudi_Demo.git
cd Gaudi_Demo
```

2. Build the Apptainer images: 

Run the commands manually or submit a batch job.

```bash
./gaudi-apptainer.sh -i <image_name.sif> -d <def_file.def> build
./cuda-apptainer.sh -i <image_name.sif> -d <def_file.def> build
-------------------------
sbatch build.sh
```

The bash files handles the installation using definition files (```gaudi-apptainer.def```, ```cuda-apptainer.def```) and package requirements (```requirements.txt```, ```requirements_nodeps.txt```, ```requirements_cuda.txt```). Running the ```build``` command will create the Apptainer images (```gaudi-apptainer.sif```, ```cuda-apptainer.sif```). 

Each repository folder provides the necessary package lists. Before building an image, make sure the definition file pulls the proper package list. You can change the Apptainer image name and the definition file used to create that image using the ```-i``` and ```-d``` commands.

3. Install custom packages (Optional):

If running your own code, you may need extra packages. Test the dependencies needed for those packages using 
```bash
pip install --dry-run <package>
```
If the package also installs NVIDIA packages (cuda, cublas, cudnn, etc.), add the package to ```requirements_nodeps.txt```. Otherwise, add the package to ```requirements.txt```. If using NVIDIA GPUs, add the package to ```requirements_cuda.txt```. NVIDIA packages will corrupt the Gaudi-specific torch packages. Repeat step 2 to rebuild the image with the new packages.

## Code Changes

Here is a list of changes that make the code compatible with Gaudi HPUs:

1. Enable Eager or Lazy mode before running a script.
```bash
EXPORT PT_HPU_LAZY_MODE = 0 #Eager
EXPORT PT_HPU_LAZY_MODE = 1 #Lazy
```

2. Change the device.
```python
device = torch.device("hpu") 
```

3. Use ```torch.compile``` to speed up training (only with Eager mode).
```python
model = torch.compile(model,backend="hpu_backend")
```

4. Move models and tensors to ```hpu``` instead of ```cuda```.
```python
model.to(device)
...
for i, (samples, targets) in enumerate(data_loader):
  samples, targets = samples.float().to(device), targets.float().to(device)
```

5. Mark steps in training/evaluation loops (for Lazy mode).
```python
import habana_frameworks.torch.core as htcore
...
for i, (samples, targets) in enumerate(data_loader):
  ...
  if lazy_mode:
    htcore.mark_step()
```

6. If using mixed precision (e.g., segmentation), use ```bfloat16``` instead of ```fp16```. ```GradScaler``` is not needed when using ```bfloat16```.
```python
with torch.autocast(device_type='hpu', dtype=torch.bfloat16, enabled=True):
  output = model(samples)
```

7. Move all tensors to ```cpu``` before saving to disk.
```python
cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
cpu_teacher_state = {k: v.cpu() for k, v in teacher.state_dict().items()}
```

8. Use Habana Collective Communications Library (HCCL) for distributed training.
```python
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
args.world_size, args.rank, args.local_rank = initialize_distributed_hpu()
args.dist_backend = 'hccl'
dist.init_process_group(args.dist_backend, rank=args.rank, world_size=args.world_size)
```

## Scripts

Gaudi provides two execution modes: Eager and Lazy. Eager mode (default) executes commands immediately, resulting in slower performance. Eager mode can be sped up with ```torch.compile```, which wraps the model into an optimized graph. Lazy mode accumulates operations into an execution graph, optimizing performance but requiring larger initial overhead. 

**[ALWAYS USE Lazy MODE] Although Lazy mode is deprecated, based on personal testing, Eager mode may lead to significant underperformance in metrics.** 

4 different code repositories are provided as a demo: Ark_Plus, BenchmarkTransferLearning, MedMNIST (2D), and MedMNIST (3D).

Run ```./apptainer.sh run``` to start an interactive session, or run ```./apptainer.sh exec <command>``` to execute a command within the container. Example bash scripts, which can be submitted as jobs, are provided in the ```scripts``` directory. Modify data and output paths in the scripts and in ```classification_config.yaml``` and ```segmentation_config.yaml```.

To use Gaudi HPUs, set ```args.device``` to ```hpu``` and use ```./gaudi-apptainer.sh``` 

To use NVIDIA GPUs, set ```args.device``` to ```cuda``` and use ```./cuda-apptainer.sh``` 

## Performance Tables

Ark+, Cyclic Training with VinDrCXR + NIHCXR14

<table>
  <thead>
    <tr>
      <th></th>
      <th colspan="2">VinDrCXR</th>
      <th colspan="2">ChestXray14</th>
    </tr>
    <tr>
      <th></th>
      <th>Student</th>
      <th>Teacher</th>
      <th>Student</th>
      <th>Teacher</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Gaudi (Lazy)</td>
      <td>94.12</td>
      <td>93.47</td>
      <td>79.80</td>
      <td>79.52</td>
    </tr>
    <tr>
      <td>Gaudi (Eager)</td>
      <td>84.82</td>
      <td>83.99</td>
      <td>70.19</td>
      <td>68.62</td>
    </tr>
    <tr>
      <td>Gaudi (Lazy + DDP)</td>
      <td>94.33</td>
      <td>93.84</td>
      <td>79.98</td>
      <td>79.67</td>
    </tr>
    <tr>
      <td>Official</td>
      <td colspan="2" align="center">94.14</td>
      <td colspan="2" align="center">79.89</td>
    </tr>
  </tbody>
</table>

BenchmarkTransferLearning, Swin-B

<table><tbody>
<th valign="bottom">Compute Mode</th>
<th valign="bottom">Dataset</th>
<th valign="bottom">AUC</th>

<tr>
<td rowspan = "4" align="center">Gaudi (Lazy)</td>
<td align="center">NIHCXR14</td>
<td align="center">81.30 ± 0.16</td>
</tr>

<tr>
<td align="center">VinDrCXR</td>
<td align="center">90.63 ± 0.37</td>
</tr>

<tr>
<td align="center">SCR-Heart</td>
<td align="center">94.42 ± 0.19</td>
</tr>

<tr>
<td align="center">SCR-Clavicle</td>
<td align="center">86.64 ± 0.60</td>
</tr>

<tr>
<td align="center">VinDrCXR</td>
<td align="center">72.14 ± 0.20</td>
</tr>
</tbody></table>

ChestMNIST, 224x224, ResNet50 

<table><tbody>
<th valign="bottom">Compute Mode</th>
<th valign="bottom">AUC</th>
<th valign="bottom">ACC</th>
<th valign="bottom">Runtime (hh:mm:ss)</th>

<tr>
<td align="center">Gaudi (Eager/Compile)</td>
<td align="center">77.4626 ± 0.5558</td>
<td align="center">94.7242 ± 0.0159</td>
<td align="center">18:38:16</td>
</tr>

<tr>
<td align="center">Gaudi (Eager/No Compile)</td>
<td align="center">77.3181 ± 0.4341</td>
<td align="center">94.7361 ± 0.0395</td>
<td align="center">23:45:05</td>
</tr>

<tr>
<td align="center">Gaudi (Lazy)</td>
<td align="center">77.2099 ± 0.3453</td>
<td align="center">94.7066 ± 0.0959</td>
<td align="center">17:17:03</td>
</tr>

<tr>
<td align="center">NVIDIA A100</td>
<td align="center">77.6912 ± 0.1632</td>
<td align="center">94.6306 ± 0.0369</td>
<td align="center">29:34:31</td>
</tr>

<tr>
<td align="center">Official</td>
<td align="center">77.3</td>
<td align="center">94.8</td>
<td align="center">N/A</td>
</tr>

</tbody></table>

NoduleMNIST3D, 28x28x28, ResNet50

<table><tbody>
<th valign="bottom">Compute Mode</th>
<th valign="bottom">AUC</th>
<th valign="bottom">ACC</th>

<tr>
<td align="center">Gaudi (Lazy)</td>
<td align="center">0.881 ± 0.4497</td>
<td align="center">0.847 ± 0.7930</td>
</tr>

<tr>
<td align="center">Official</td>
<td align="center">0.875</td>
<td align="center">0.847</td>
</tr>

</tbody></table>


## Known Limitations/Issues

Official issues are provided in Habana's documentation: [Known Issues](https://docs.habana.ai/en/latest/Release_Notes/GAUDI_Release_Notes.html#known-issues-and-limitations-v1-24-0)

Check the official documentation for updates.

Here are the issues and limitations I have encountered:

1) Using Eager mode can lead to ~10% decrease in metrics, even though no error is thrown. Use Lazy mode at all times to ensure proper model performance. Lazy mode should generally be faster too.

2) Lazy mode may require more memory to account for the compiled graphs.

3) Operations not supported by Gaudi will fall back to CPU, which may affect performance.

4) Gaudi's ```torch.compile``` currently does not support many operations, despite being Intel's preferred execution mode. As a result, many scripts cannot use this execution mode. 

5) ```synStatus=8 [Device not found]```: When interrupted, Gaudi processes may not exit gracefully. To run a script afterwards, you must kill the processes manually (use hl-smi to find process IDs).

6) ```torchrun``` is not implemented in Gaudi. Use ```python -m torch.distributed.run``` instead.

7) Do not use the ```device_ids``` parameter with DistributedDataParallel models, as initialization with HPU differs from torch/cuda.

8) When using mixed precision, torch casts operations to certain data types (lower vs higher precision. These data types differ when using Gaudi compared to NVIDIA chips. The default data types for Gaudi are provided in the [documentation](https://docs.habana.ai/en/latest/PyTorch/PyTorch_Mixed_Precision/index.html#registered-operators). 

9) Gaudi doesn’t support ```SyncBatchNorm```, which is currently strictly a GPU operation. This may lead to degraded performance when performing multi-HPU training with small batch sizes. Use larger batch sizes if possible, or switch to transformer models that do not use batch norm.

10) Gaudi seems less memory-efficient when processing 3D data. However, more experimentation is required to confirm this.

## Acknowledgements

The base code was cloned from the official Ark, BenchmarkTransferLearning, and MedMNIST repositories: [Ark](https://github.com/jlianglab/Ark/tree/main), [BenchmarkTransferLearning](https://github.com/MR-HosseinzadehTaher/BenchmarkTransferLearning/tree/main), [MedMNIST Library](https://github.com/MedMNIST/MedMNIST), and [MedMNIST Experiments](https://github.com/MedMNIST/experiments). I utilize implementations of Swin Transformer and U-Net from official repositories: [Swin Transformer](https://github.com/microsoft/Swin-Transformer) and [segmentation_models.pytorch](https://github.com/qubvel-org/segmentation_models.pytorch) I adapted the code to run on Gaudi. Necessary changes were applied from the official [Gaudi documentation](https://docs.habana.ai/en/latest/PyTorch/index.html) and from [Habana Labs' reference repository](https://github.com/HabanaAI/Model-References/tree/master/PyTorch/computer_vision/classification/torchvision). The Apptainer scripts were adapted from Jae Yul Shin's code. Special thanks to the ASU Research Computing team for help with using and debugging the Gaudi HPUs, Benjamin Malamuth for discovering degraded metric performance with Eager mode, and Prof. Jianming Liang for guidance on experiments. This work utilized NVIDIA GPUs and Intel Gaudi2 HPUs provided by ASU Research Computing through the Sol Supercomputer.
