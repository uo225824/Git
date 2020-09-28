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

imgdir_path = pathlib.Path('D:\Python\Git\Tema 13\cat_dog_images')

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

labels = [1 if 'dog' in os.path.basename(file) else 0
          for file in file_list]
print(labels)



ds_files_labels = tf.data.Dataset.from_tensor_slices(
    (file_list, labels))

for item in ds_files_labels:
    print(item[0].numpy(), item[1].numpy())


def load_and_preprocess(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [img_height, img_width])
    image /= 255.0

    return image, label


img_width, img_height = 120, 80

ds_images_labels = ds_files_labels.map(load_and_preprocess)

fig = plt.figure(figsize=(10, 5))
for i, example in enumerate(ds_images_labels):
    print(example[0].shape, example[1].numpy())
    ax = fig.add_subplot(2, 3, i + 1)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(example[0])
    ax.set_title('{}'.format(example[1].numpy()),
                 size=15)

plt.tight_layout()
# plt.savefig('ch13-catdog-dataset.pdf')
plt.show()


import tensorflow_datasets as tfds
print(len(tfds.list_builders()))
print(tfds.list_builders()[:5])

celeba_bldr = tfds.builder('celeb_a')

# Download the data, prepare it, and write it to disk
celeba_bldr.download_and_prepare()

# Load data from disk as tf.data.Datasets

datasets = celeba_bldr.as_dataset(shuffle_files=False)

datasets.keys()


#import tensorflow as tf
ds_train = datasets['train']
assert isinstance(ds_train, tf.data.Dataset)

example = next(iter(ds_train))
print(type(example))
print(example.keys())


ds_train = ds_train.map(lambda item:
     (item['image'], tf.cast(item['attributes']['Male'], tf.int32)))

ds_train = ds_train.batch(18)
images, labels = next(iter(ds_train))

print(images.shape, labels)

fig = plt.figure(figsize=(12, 8))
for i, (image, label) in enumerate(zip(images, labels)):
    ax = fig.add_subplot(3, 6, i + 1)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(image)
    ax.set_title('{}'.format(label), size=15)

plt.show()


mnist, mnist_info = tfds.load('mnist', with_info=True,
                              shuffle_files=False)

print(mnist_info)

print(mnist.keys())

#C:\Users\Christian\tensorflow_datasets

ds_train = mnist['train']

assert isinstance(ds_train, tf.data.Dataset)

ds_train = ds_train.map(lambda item:
                        (item['image'], item['label']))

ds_train = ds_train.batch(10)
batch = next(iter(ds_train))
print(batch[0].shape, batch[1])

fig = plt.figure(figsize=(15, 6))
for i, (image, label) in enumerate(zip(batch[0], batch[1])):
    ax = fig.add_subplot(2, 5, i + 1)
    ax.set_xticks([]);
    ax.set_yticks([])
    ax.imshow(image[:, :, 0], cmap='gray_r')
    ax.set_title('{}'.format(label), size=15)

plt.show()


#Building an NN model in TensorFlow

##Building a linear regression model with keras

X_train = np.arange(10).reshape((10, 1))
y_train = np.array([1.0, 1.3, 3.1,
                    2.0, 5.0, 6.3,
                    6.6, 7.4, 8.0,
                    9.0])


plt.plot(X_train, y_train, 'o', markersize=10)
plt.xlabel('x')
plt.ylabel('y')
plt.show()




X_train_norm = (X_train - np.mean(X_train))/np.std(X_train)

ds_train_orig = tf.data.Dataset.from_tensor_slices(
    (tf.cast(X_train_norm, tf.float32),
     tf.cast(y_train, tf.float32)))



class MyModel(tf.keras.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.w = tf.Variable(0.0, name='weight')
        self.b = tf.Variable(0.0, name='bias')

    def call(self, x):
        return self.w*x + self.b


model = MyModel()

model.build(input_shape=(None, 1))
model.summary()



def loss_fn(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_true - y_pred))


## testing the function:
yt = tf.convert_to_tensor([1.0])
yp = tf.convert_to_tensor([1.5])

loss_fn(yt, yp)


def train(model, inputs, outputs, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss_fn(model(inputs), outputs)
    dW, db = tape.gradient(current_loss, [model.w, model.b])
    model.w.assign_sub(learning_rate * dW)
    model.b.assign_sub(learning_rate * db)


tf.random.set_seed(1)

num_epochs = 200
log_steps = 100
learning_rate = 0.001
batch_size = 1
steps_per_epoch = int(np.ceil(len(y_train) / batch_size))


ds_train = ds_train_orig.shuffle(buffer_size=len(y_train))
ds_train = ds_train.repeat(count=None)
ds_train = ds_train.batch(1)

Ws, bs = [], []

for i, batch in enumerate(ds_train):
    if i >= steps_per_epoch * num_epochs:
        break
    Ws.append(model.w.numpy())
    bs.append(model.b.numpy())

    bx, by = batch
    loss_val = loss_fn(model(bx), by)

    train(model, bx, by, learning_rate=learning_rate)
    if i%log_steps==0:
        print('Epoch {:4d} Step {:2d} Loss {:6.4f}'.format(
              int(i/steps_per_epoch), i, loss_val))


print('Final Parameters:', model.w.numpy(), model.b.numpy())


X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))


fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training examples', 'Linear Reg.'], fontsize=15)
ax.set_xlabel('x', size=15)
ax.set_ylabel('y', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['Weight w', 'Bias unit b'], fontsize=15)
ax.set_xlabel('Iteration', size=15)
ax.set_ylabel('Value', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
#plt.savefig('ch13-linreg-1.pdf')

plt.show()


##Model training via the .compile() and .fit() methods

tf.random.set_seed(1)
model = MyModel()
#model.build((None, 1))

model.compile(optimizer='sgd',
              loss=loss_fn,
              metrics=['mae', 'mse'])

model.fit(X_train_norm, y_train,
          epochs=num_epochs, batch_size=batch_size,
          verbose=1)

print(model.w.numpy(), model.b.numpy())


X_test = np.linspace(0, 9, num=100).reshape(-1, 1)
X_test_norm = (X_test - np.mean(X_train)) / np.std(X_train)

y_pred = model(tf.cast(X_test_norm, dtype=tf.float32))


fig = plt.figure(figsize=(13, 5))
ax = fig.add_subplot(1, 2, 1)
plt.plot(X_train_norm, y_train, 'o', markersize=10)
plt.plot(X_test_norm, y_pred, '--', lw=3)
plt.legend(['Training Samples', 'Linear Regression'], fontsize=15)

ax = fig.add_subplot(1, 2, 2)
plt.plot(Ws, lw=3)
plt.plot(bs, lw=3)
plt.legend(['W', 'bias'], fontsize=15)
plt.show()

#Building a multilayer perceptron for classifying flowers in the Iris dataset

iris, iris_info = tfds.load('iris', with_info=True)

print(iris_info)

tf.random.set_seed(1)

ds_orig = iris['train']
ds_orig = ds_orig.shuffle(150, reshuffle_each_iteration=False)

print(next(iter(ds_orig)))

ds_train_orig = ds_orig.take(100)
ds_test = ds_orig.skip(100)

ds_train_orig = ds_train_orig.map(
    lambda x: (x['features'], x['label']))

ds_test = ds_test.map(
    lambda x: (x['features'], x['label']))

next(iter(ds_train_orig))

##NN with 2 layers

iris_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation='sigmoid',
                          name='fc1', input_shape=(4,)),
    tf.keras.layers.Dense(3, name='fc2', activation='softmax')])

iris_model.summary()


iris_model.compile(optimizer='adam',
                   loss='sparse_categorical_crossentropy',
                   metrics=['accuracy'])


num_epochs = 100
training_size = 100
batch_size = 2
steps_per_epoch = np.ceil(training_size / batch_size)

ds_train = ds_train_orig.shuffle(buffer_size=training_size)
ds_train = ds_train.repeat()
ds_train = ds_train.batch(batch_size=batch_size)
ds_train = ds_train.prefetch(buffer_size=1000)


history = iris_model.fit(ds_train, epochs=num_epochs,
                         steps_per_epoch=steps_per_epoch,
                         verbose=0)

hist = history.history

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1)
ax.plot(hist['loss'], lw=3)
ax.set_title('Training loss', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)

ax = fig.add_subplot(1, 2, 2)
ax.plot(hist['accuracy'], lw=3)
ax.set_title('Training accuracy', size=15)
ax.set_xlabel('Epoch', size=15)
ax.tick_params(axis='both', which='major', labelsize=15)
plt.tight_layout()
#plt.savefig('ch13-cls-learning-curve.pdf')

plt.show()


#Evaluating the trained model on the test dataset


results = iris_model.evaluate(ds_test.batch(50), verbose=0)
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))


##Saving and reloading the trained model

iris_model.save('iris-classifier.h5',
                overwrite=True,
                include_optimizer=True,
                save_format='h5')


iris_model_new = tf.keras.models.load_model('iris-classifier.h5')

iris_model_new.summary()

results = iris_model_new.evaluate(ds_test.batch(33), verbose=0)
print('Test loss: {:.4f}   Test Acc.: {:.4f}'.format(*results))

#try