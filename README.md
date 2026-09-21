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
./gaudi-apptainer.sh build
./cuda-apptainer.sh build
-------------------------
sbatch build.sh
```

The bash files handles the installation using definition files (```gaudi-apptainer.def```, ```cuda-apptainer.def```) and package requirements (```requirements.txt```, ```requirements_nodeps.txt```, ```requirements_cuda.txt```). Running the ```build``` command will create the Apptainer images (```gaudi-apptainer.sif```, ```cuda-apptainer.sif```).

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

4. Mark steps in training/evaluation loops (for Lazy mode).
```python
import habana_frameworks.torch.core as htcore
...
htcore.mark_step()
```

5. Move all tensors to cpu before saving to disk.
```python
cpu_model_state = {k: v.cpu() for k, v in model.state_dict().items()}
cpu_teacher_state = {k: v.cpu() for k, v in teacher.state_dict().items()}
```

6. Use Habana Collective Communications Library (HCCL) for distributed training.
```python
from habana_frameworks.torch.distributed.hccl import initialize_distributed_hpu
args.world_size, args.rank, args.local_rank = initialize_distributed_hpu()
args.dist_backend = 'hccl'
dist.init_process_group(args.dist_backend, rank=args.rank, world_size=args.world_size)
```

## Scripts

Gaudi provides two execution modes: Eager and Lazy. Eager mode (default) executes commands immediately, resulting in slower performance. Eager mode can be sped up with ```torch.compile```, which wraps the model into an optimized graph. Lazy mode accumulates operations into an execution graph, optimizing performance but requiring larger initial overhead. 

**[ALWAYS USE Lazy MODE] Although Lazy mode is deprecated, based on personal testing, Eager mode may lead to significant underperformance in metrics.** 

4 different code repositories are provided as a demo: Ark_Plus, BenchmarkTransformers, MedMNIST (2D), and MedMNIST (3D). BenchmarkTransformers only contains NIHCXR14, and it only supports distributed training (you can still use a single HPU).

Example bash scripts are provided in the ```scripts``` directory. Modify data and output paths in the scripts and in ```config.yaml```. You can run each program by submitting a job or running the commands in an interactive session (```./apptainer.sh run```).

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
      <td>0.9412</td>
      <td>0.9347</td>
      <td>0.7980</td>
      <td>0.7952</td>
    </tr>
    <tr>
      <td>Gaudi (Eager)</td>
      <td>0.8482</td>
      <td>0.8399</td>
      <td>0.7019</td>
      <td>0.6862</td>
    </tr>
    <tr>
      <td>Gaudi (Lazy + DDP)</td>
      <td>0.9433</td>
      <td>0.9384</td>
      <td>0.7998</td>
      <td>0.7967</td>
    </tr>
    <tr>
      <td>Official</td>
      <td colspan="2" align="center">0.9414</td>
      <td colspan="2" align="center">0.7989</td>
    </tr>
  </tbody>
</table>

BenchmarkTransformers, NIHCXR14, Swin-B

<table><tbody>
<th valign="bottom">Compute Mode</th>
<th valign="bottom">AUC</th>

<tr>
<td align="center">Gaudi (Lazy)</td>
<td align="center">0.8162</td>
</tr>

<tr>
<td align="center">Official</td>
<td align="center">0.8173</td>
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
<td align="center">77.4626</td>
<td align="center">94.7242</td>
<td align="center">18:38:16</td>
</tr>

<tr>
<td align="center">Gaudi (Eager/No Compile)</td>
<td align="center">77.3181</td>
<td align="center">94.7361</td>
<td align="center">23:45:05</td>
</tr>

<tr>
<td align="center">Gaudi (Lazy)</td>
<td align="center">77.2099</td>
<td align="center">94.7066</td>
<td align="center">17:17:03</td>
</tr>

<tr>
<td align="center">NVIDIA A100</td>
<td align="center">77.6912</td>
<td align="center">94.6306</td>
<td align="center">29:34:31</td>
</tr>
</tbody></table>

NoduleMNIST3D, 28x28x28, ResNet50

<table><tbody>
<th valign="bottom">Compute Mode</th>
<th valign="bottom">AUC</th>
<th valign="bottom">ACC</th>

<tr>
<td align="center">Gaudi (Lazy)</td>
<td align="center">0.881</td>
<td align="center">0.847</td>
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

3) Gaudi's ```torch.compile``` currently does not support many operations, despite being Intel's preferred execution mode. As a result, many scripts cannot use this execution mode. 

4) ```synStatus=8 [Device not found]``` When interrupted, Gaudi processes may not exit gracefully. To run a script afterwards, you must kill the processes manually (use hl-smi to find process IDs).

5) ```torchrun``` is not implemented in Gaudi. Use ```python -m torch.distributed.run``` instead.

6) Do not use the ```device_ids``` parameter with DistributedDataParallel models, as initialization with HPU differs from torch/cuda.

7) Gaudi doesn’t support ```SyncBatchNorm```, which is currently strictly a GPU operation. This may lead to degraded performance when performing multi-HPU training with small batch sizes. Use larger batch sizes if possible, or switch to transformer models that do not use batch norm.

8) Gaudi seems to be less memory-efficient when processing 3D data. However, more experimentation is required to confirm this.

## Acknowledgements

The base code was cloned from the official Ark, BenchmarkTransformers, and MedMNIST repositories: [Ark](https://github.com/jlianglab/Ark/tree/main), [BenchmarkTransformers](https://github.com/jlianglab/BenchmarkTransformers/tree/main), [MedMNIST Library](https://github.com/MedMNIST/MedMNIST), and [MedMNIST Experiments](https://github.com/MedMNIST/experiments). I adapted the code to run on Gaudi. The Apptainer scripts were adapted from Jae Yul Shin's code. Special thanks to the ASU Research Computing team for help with using and debugging the Gaudi HPUs, Benjamin Malamuth for discovering degraded metric performance with Eager mode, and Prof. Jianming Liang for guidance on experiments. This work utilized NVIDIA GPUs and Intel Gaudi2 HPUs provided by ASU Research Computing through the Sol Supercomputer.
