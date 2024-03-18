# 并行策略



# Pytorch 并行方法

## torch.nn.DataParallel

> torch.nn.DataParallel(*module*, *device_ids=None*, *output_device=None*, *dim=0*)
>
> * **module** ([*Module*](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) – module to be parallelized
> * **device_ids** ([*list*](https://docs.python.org/3/library/stdtypes.html#list) *of* [*int*](https://docs.python.org/3/library/functions.html#int) *or* [*torch.device*](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – CUDA devices (default: all devices)
> * **output_device** ([*int*](https://docs.python.org/3/library/functions.html#int) *or* [*torch.device*](https://pytorch.org/docs/stable/tensor_attributes.html#torch.device)) – device location of output (default: device_ids[0])

顾名思义，``torch.nn.DataParallel`` 是(仅)支持数据并行的一种策略：把模型复制到 n 个 GPU 上，然后把一个批次的数据自动拆分到 n 个 GPU 上 (batch_size = 30, n_gpu = 3 => [10, ], [10, ], [10, ])

具体原理：采用**单进程多线程策略**，网络在**前向传播**的时候会将 model 从主卡(默认是逻辑0卡)广播到所有的 device 上，input_data 会在 batch 这个维度被分组后 upload 到不同的 device 上计算，模型的输出会被 gather 到 output_device 上计算 loss，然后把 loss scatter 到每个 gpu 上，每个 gpu 再通过反向传播计算得到梯度。**反向传播**时，每个卡上的梯度会 reduce 到主卡上，求得梯度的均值后，主设备的模型参数就可以得到更新了，下一轮的前向传播开始时，主卡就会把新的模型参数广播到所有设备，循环上述过程。

**优点：**使用非常简单，只需要把需要模型用 torch.nn.DataParallel 包裹起来就可以了

**缺点：**

1. 采用单进程多线程模型，受到 GIL (Global Iterpreter Lock) 的限制很严重
2. 节点负载不均衡，master 节点的计算任务、通讯量很重 (每一个 batch 的计算都要发送一遍网络参数)，从而导致网络阻塞，降低训练速度 (加速效果明显小于 DDP)
3. 仅支持单机多卡，仅支持数据并行策略，不支持模型并行

**官方现已经不推荐使用该方法**

## torch.nn.parallel.DistributedDataParallel

> torch.nn.parallel.DistributedDataParallel(*module*, *device_ids=None*, *output_device=None*, *dim=0*, *broadcast_buffers=True*, *process_group=None*, *bucket_cap_mb=25*, *find_unused_parameters=False*, *check_reduction=False*, *gradient_as_bucket_view=False*, *static_graph=False*, *delay_all_reduce_named_params=None*, *param_to_hook_all_reduce=None*, *mixed_precision=None*, *device_mesh=None*)

多进程策略，支持单机多卡，多机多卡，同时支持数据并行和模型并行两种策略，支持混合精度训练

### **DistributedDataParallel 和 DataParallel 区别：**

1. DDP 使用多进程策略，避免了单进程多线程策略遇到的 GIL 限制
2. 参数更新方式：DDP 预先在各个设备上单独 load 模型，在各进程梯度计算完成之后，各进程通过 all_reduce (内部具体实现的是 ring all reduce) 通信汇总平均得到全局梯度，**各进程用该梯度来独立的更新参数**；而 DP 仅有主设备获得了全局梯度并更新模型参数，在下一轮 batch 计算前需要将模型参数广播到其他设备上，通信开销巨大
3. DDP 还支持单机多卡、多机多卡、模型并行策略 (当 DDP 使用模型并行策略时，一个进程控制多个 GPU，此时单个进程就采用的是 DP 策略)

### **使用注意：**

1. 模型要用 DistributedDataParallel 包裹起来，包裹起来后 model 就变成了 ddp_module，真实的模型其实是 `model.module`，在保存模型的时候注意要保存 `model.module` 的参数 (如果忘记了 load 进来的就是 ddp_module，使用的时候要注意)

   ```python
   from torch.nn.parallel import DistributedDataParallel as DDP
   
   model = DDP(model, device_ids=[local_rank])
   ```

2. 设置 `DistributedSampler` 来打乱数据，因为一个batch被分配到了好几个进程中，要确保不同的GPU拿到的不是同一份数据

   ```python
   from torch.utils.data DataLoader
   from torch.utils.data.distributed import DistributedSampler
   
   train_sampler = DistributedSampler(train_dataset, shuffle=True)
   # DataLoader中的shuffle应该设置为False（默认），因为打乱的任务交给了sampler
   train_loader = DataLoader(train_dataset, batch_size=bz, sampler=train_sampler)
   ```

3. 要告诉每个进程自己的 id，即使用哪一块 GPU (其实就是 local_rank)

   ```python
   # local_rank 在 pytorch 1.x 和 2.x 有所不同
   # DDP 在 pytorch 1.x 启动为 python -m torch.distributed.launch --nnodes 1 --nproc_per_node 4 script.py
   # DDP 在 pytorch 2.x 启动为 torchrun --nnodes 1 --nproc_per_node 4 script.py
   # 上面两种方式获取 local_rank 的方式是不一样的
   
   # pytorch 1.x python -m torch.distributed.launch 启动
   # 会自动传入用命令行参数传入一个 local_rank, 需要手动接收
   parser = argparse.ArgumentParser()
   parser.add_argument("--local_rank", default=-1, type=int)
   args = parser.parse_args()
   local_rank = args.local_rank
   
   # pytorch 2.x torchrun 启动
   # local_rank 应当使用环境变量获取
   import os
   local_rank = int(os.environ["LOCAL_RANK"])
   
   # 如果使用 Transformers 的 HfArgumentParser 和 TrainingArguments, 那么获取方法就非常统一
   parser = HfArgumentParser((TrainingArguments, ))
   args: TrainingArguments = parser.parse_args_into_dataclasses()[0]
   local_rank = args.local_rank
   ```

4. 如果需要做BatchNormalization，需要对数据进行同步（还待研究，**挖坑**）

### **torch.multiprocessing.spawn**

官方启动 DDP 是使用 torch.multiprocessing.spawn 开启多进程

>torch.multiprocessing.spawn(
>    fn,
>    args=(),
>    nprocs=1,
>    join=True,
>    daemon=False,
>    start_method='spawn',
>)
>
>* **fn** (*function*) –函数被称为派生进程的入口点。必须在模块的顶层定义此函数，以便对其进行pickle和派生。这是多进程强加的要求。该函数称为`fn(i, *args)`，其中`i`是进程索引，`args`是传递的参数元组。
>* **args** (*tuple*) – 传递给 `fn` 的参数.
>* **nprocs** (*int*) – 派生的进程数.
>* **join** (*bool*) – 执行一个阻塞的join对于所有进程.
>* **daemon** (*bool*) – 派生进程守护进程标志。如果设置为True，将创建守护进程.
>
>其中，`fn` 是要在子进程中运行的函数，`args` 是传递给该函数的参数，`nprocs` 是要启动的进程数。当 `nprocs` 大于 1 时，会创建多个子进程，并在每个子进程中调用 `fn` 函数，每个子进程都会使用不同的进程 ID 进行标识。当 `nprocs` 等于 1 时，会在当前进程中直接调用 `fn` 函数，而不会创建新的子进程。

在上面提到的代码中，`torch.multiprocessing.spawn` 函数的具体调用方式如下：

```python
torch.multiprocessing.spawn(process_fn, args=(parsed_args,), nprocs=world_size)
```

其中，`process_fn` 是要在子进程中运行的函数，`args` 是传递给该函数的参数，`nprocs` 是要启动的进程数，即推断出的 GPU 数量。这里的 `process_fn` 函数应该是在其他地方定义的，用于执行具体的训练任务。在多进程编程中，每个子进程都会调用该函数来执行训练任务。

需要注意的是，`torch.multiprocessing.spawn` 函数会自动将数据分布到各个进程中，并在所有进程执行完成后进行同步，以确保各个进程之间的数据一致性。同时，该函数还支持多种进程间通信方式，如共享内存（Shared Memory）、管道（Pipe）等，可以根据具体的需求进行选择。

给予`process_fn` 函数如下：

```python
def process_fn(rank, args):
    local_args = copy.copy(args)
    local_args.local_rank = rank
    main(local_args)
```

其中，`rank` 参数是当前子进程的 ID，由 `torch.multiprocessing.spawn` 函数**自动分配**。而 `args` 参数是在调用 `torch.multiprocessing.spawn` 函数时传递的，其值为 `(parsed_args,)`，表示 `args` 是一个元组，其中包含了一个元素 `parsed_args`。

### 完整例子

```python
################
## main.py文件
import argparse
from tqdm import tqdm
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
# 新增：
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

### 1. 基础模块 ### 
# 假设我们的模型是这个，与DDP无关
class ToyModel(nn.Module):
    def __init__(self):
        super(ToyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 假设我们的数据是这个
def get_dataset():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    my_trainset = torchvision.datasets.CIFAR10(root='./data', train=True, 
        download=True, transform=transform)
    # DDP：使用DistributedSampler，DDP帮我们把细节都封装起来了。
    #      用，就完事儿！sampler的原理，第二篇中有介绍。
    train_sampler = torch.utils.data.distributed.DistributedSampler(my_trainset)
    # DDP：需要注意的是，这里的batch_size指的是每个进程下的batch_size。
    #      也就是说，总batch_size是这里的batch_size再乘以并行数(world_size)。
    trainloader = torch.utils.data.DataLoader(my_trainset, 
        batch_size=16, num_workers=2, sampler=train_sampler)
    return trainloader
    
### 2. 初始化我们的模型、数据、各种配置  ####
# DDP：从外部得到local_rank参数
parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=-1, type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank

# DDP：DDP backend初始化
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

# 准备数据，要在DDP初始化之后进行
trainloader = get_dataset()

# 构造模型
model = ToyModel().to(local_rank)
# DDP: Load模型要在构造DDP模型之前，且只需要在master上加载就行了。
ckpt_path = None
if dist.get_rank() == 0 and ckpt_path is not None:
    model.load_state_dict(torch.load(ckpt_path))
# DDP: 构造DDP model
model = DDP(model, device_ids=[local_rank], output_device=local_rank)

# DDP: 要在构造DDP model之后，才能用model初始化optimizer。
optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# 假设我们的loss是这个
loss_func = nn.CrossEntropyLoss().to(local_rank)

### 3. 网络训练  ###
model.train()
iterator = tqdm(range(100))
for epoch in iterator:
    # DDP：设置sampler的epoch，
    # DistributedSampler需要这个来指定shuffle方式，
    # 通过维持各个进程之间的相同随机数种子使不同进程能获得同样的shuffle效果。
    trainloader.sampler.set_epoch(epoch)
    # 后面这部分，则与原来完全一致了。
    for data, label in trainloader:
        data, label = data.to(local_rank), label.to(local_rank)
        optimizer.zero_grad()
        prediction = model(data)
        loss = loss_func(prediction, label)
        loss.backward()
        iterator.desc = "loss = %0.3f" % loss
        optimizer.step()
    # DDP:
    # 1. save模型的时候，和DP模式一样，有一个需要注意的点：保存的是model.module而不是model。
    #    因为model其实是DDP model，参数是被`model=DDP(model)`包起来的。
    # 2. 只需要在进程0上保存一次就行了，避免多次保存重复的东西。
    if dist.get_rank() == 0:
        torch.save(model.module.state_dict(), "%d.ckpt" % epoch)


################
## Bash运行
# DDP: 使用torch.distributed.launch启动DDP模式
# 使用CUDA_VISIBLE_DEVICES，来决定使用哪些GPU
# CUDA_VISIBLE_DEVICES="0,1" python -m torch.distributed.launch --nproc_per_node 2 main.py
```

