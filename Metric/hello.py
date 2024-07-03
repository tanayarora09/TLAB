import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.v2 as transforms

print(torch.cuda.is_available())

print(torch.backends.cudnn.is_available())

print(torch.backends.cudnn.allow_tf32)

"""
def get_cifar():
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    @tf.function
    def simple_preprocess(img, label):
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [224, 224]) / 255.0
        #img  = (img - [0.4914, 0.4822, 0.4465]) / [0.2023, 0.1994, 0.2010]
        return img, tf.one_hot(label, 10)
    dt, dv = tfds.load('cifar10', split=['train', 'test'], as_supervised = True, shuffle_files=True)
    dt = dt.cache().map(simple_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(128).with_options(options)
    dv = dv.cache().map(simple_preprocess, num_parallel_calls=tf.data.AUTOTUNE).batch(128).with_options(options)
    return dt, dv
"""

batch_size = 64

train_transforms = transforms.Compose(
    [transforms.Resize([224,]),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale = True),   
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010)),
    transforms.RandomCrop([200,]),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15), 
    transforms.Resize([224,]),
    ]
)

one_hot_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.int64)), 
    transforms.Lambda(lambda x: F.one_hot(x, num_classes=10).float()) 
])

test_transforms = transforms.Compose(
    [transforms.Resize(224),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale = True),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                        (0.2023, 0.1994, 0.2010))
    ])



train_data = torchvision.datasets.CIFAR10("../DATA/", train = True, download = True,
                                        transform = train_transforms, target_transform = one_hot_transform)

test_data = torchvision.datasets.CIFAR10("../DATA/", train = False, download = True,
                                        transform = test_transforms, target_transform = one_hot_transform)

dt = torch.utils.data.DataLoader(train_data, batch_size = batch_size,  
                                num_workers = 4, pin_memory = True,
                                prefetch_factor = 4, pin_memory_device = "cuda")

dv = torch.utils.data.DataLoader(test_data, batch_size = batch_size,  
                                num_workers = 4, pin_memory = True,
                                prefetch_factor = 4, pin_memory_device = "cuda")


for step, (x, y) in enumerate(dt):
    if step == 0:
        print(x.is_cuda)
        print(x.is_contiguous())
        print(x.shape)
        print(y.shape)
        print(y)
    if step < 3:
        print(x[:10, 0, 0, 0])
    if step % 100 == 0:
        print(step)

for step, (x, y) in enumerate(dt):
    if step == 0:
        print(x.is_cuda)
        print(x.is_contiguous())
        print(x.shape)
        print(y.shape)
        print(y)
    if step < 3:
        print(x[:10, 0, 0, 0])
    if step % 100 == 0:
        print(step)