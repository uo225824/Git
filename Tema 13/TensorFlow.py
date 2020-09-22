#Creating tensors in TensoFlow

import tensorflow as tf
import numpy as np


a = np.array([1, 2, 3], dtype=np.int32)
b = [4, 5, 6]

t_a = tf.convert_to_tensor(a)
t_b = tf.convert_to_tensor(b)

print(t_a)
print(t_b)

tf.is_tensor(a), tf.is_tensor(t_a)

t_ones = tf.ones((2, 3))

t_ones.shape

t_ones.numpy()

const_tensor = tf.constant([1.2, 5, np.pi], dtype=tf.float32)

print(const_tensor)

#Manipulating the data type and shape of a tensor


##tf.cast() change the data type of a tensor to a desired type:

t_a_new=tf.cast(t_a,tf.int64)

##Transposing a tensor

t=tf.random.uniform(shape=(3,5))
t_tr=tf.transpose(t)

##Reshaping a tensor

t=tf.zeros((30,))
t_reshape=tf.reshape(t,shape=(5,6))

##Removing the unnecessary dimensions

t=tf.zeros((1,2,1,4,1))
t_sqz=tf.squeeze(t,axis=(2,4))



#Applying mathematical operations to tensors

tf.random.set_seed(1)
t1=tf.random.uniform(shape=(5,2),minval=-1.0,maxval=1.0)
t2=tf.random.normal(shape=(5,2),mean=0.0,stddev=1.0)

t3=tf.multiply(t1,t2).numpy()

##mean, sum, and standard deviation along a certain axis we
##can use tf.math.reduce_mean(), tf.math.reduce_sum(), and tf.math.reduce_std()

t4=tf.math.reduce_mean(t1,axis=0)

##t1Xt2'

t5=tf.linalg.matmul(t1,t2,transpose_b=True)


##LP norm

norm_t1=tf.norm(t1,ord=2,axis=1).numpy()


#Split, stack, and concatenate tensors


##Providing the number of splits (must be divisible):


tf.random.set_seed(1)
t=tf.random.uniform((6,))
t_split=tf.split(t,num_or_size_splits=3)
[item.numpy() for item in t_split]

t_split[1].numpy()

##Providing the sizes of different splits:

tf.random.set_seed(1)
t=tf.random.uniform((5,))

t_split=tf.split(t, num_or_size_splits=[3,2])

[item.numpy() for item in t_split]

##Concatenate

A=tf.ones((3,))
B=tf.zeros((2,))
C=tf.concat([A,B],axis=0)

##Stack


A=tf.ones((3,))
B=tf.zeros((3,))
C=tf.stack([A,B],axis=1)


#Creating a TensorFlow Dataset from existing tensors

a=[1.2, 3.4, 7.5, 4.1, 5.0, 1.0]
ds=tf.data.Dataset.from_tensor_slices(a)

for item in ds:
    print(item)

##Create batches from a dataset


ds_bathc=ds.batch(3)

for i, elem in enumerate(ds_bathc, 1):
    print('batch {}:' .format(i), elem.numpy())

#Combining two tensors into a joint dataset

tf.random.set_seed(1)
t_x = tf.random.uniform([4, 3], dtype=tf.float32)
t_y = tf.range(4)

ds_x=tf.data.Dataset.from_tensor_slices(t_x)
ds_y=tf.data.Dataset.from_tensor_slices(t_y)

ds_joint=tf.data.Dataset.zip((ds_x,ds_y))
for example in ds_joint:
    print(' x:', example[0].numpy(), ' y:', example[1].numpy())

###Equivalent

ds_joint = tf.data.Dataset.from_tensor_slices((t_x, t_y))

#Shuffle, batch, and repeat

tf.random.set_seed(1)
ds=ds_joint.shuffle(buffer_size=len(t_x))
for example in ds:
    print('x:', example[0].numpy(),
          'y:', example[1].numpy())


ds = ds_joint.batch(batch_size=3,drop_remainder=False)
batch_x, batch_y = next(iter(ds))
print('Batch-x:\n', batch_x.numpy())


ds = ds_joint.batch(3).repeat(count=2)
for i,(batch_x, batch_y) in enumerate(ds):
    print(i, batch_x.shape, batch_y.numpy())


#Creating a dataset from files on your local storage disk


import pathlib

imgdir_path = pathlib.Path('cat_dog_images')

file_list = sorted([str(path) for path in imgdir_path.glob('*.jpg')])

print(file_list)

import matplotlib.pyplot as plt
import os

fig = plt.figure(figsize=(10, 5))
for i, file in enumerate(file_list):
    img_raw = tf.io.read_file(file)
    img = tf.image.decode_image(img_raw)
    print('Image shape: ', img.shape)
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(img)
    ax.set_title(os.path.basename(file), size=15)

# plt.savefig('ch13-catdot-examples.pdf')
plt.tight_layout()
plt.show()