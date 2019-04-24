import tensorflow as tf
from tensorflow.python.client import device_lib
import math
import subprocess

def augmentImages(images,
            resize=None, # (width, height) tuple or None
            horizontal_flip=False,
            vertical_flip=False,
            translate = 0,
            rotate=0, # Maximum rotation angle in degrees
            crop_probability=0, # How often we do crops
            crop_min_percent=0.6, # Minimum linear dimension of a crop
            crop_max_percent=1.,  # Maximum linear dimension of a crop
            mixup=0):  # Mixup coeffecient, see https://arxiv.org/abs/1710.09412.pdf
  if resize is not None:
    images = tf.image.resize_bilinear(images, resize)

  # My experiments showed that casting on GPU improves training performance
  #print(images.dtype)
  '''
  if images.dtype != tf.float32:
    images = tf.image.convert_image_dtype(images, dtype=tf.float32)
    images = tf.subtract(images, 0.5)
    images = tf.multiply(images, 2.0)
  #labels = tf.to_float(labels)
  '''
  with tf.name_scope('augmentation'):
    shp = tf.shape(images)
    batch_size, height, width = shp[0], shp[1], shp[2]
    width = tf.cast(width, tf.float32)
    height = tf.cast(height, tf.float32)

    # The list of affine transformations that our image will go under.
    # Every element is Nx8 tensor, where N is a batch size.
    transforms = []
    identity = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
    if horizontal_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [-1., 0., width, 0., 1., 0., 0., 0.], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if vertical_flip:
      coin = tf.less(tf.random_uniform([batch_size], 0, 1.0), 0.5)
      flip_transform = tf.convert_to_tensor(
          [1, 0, 0, 0, -1, height, 0, 0], dtype=tf.float32)
      transforms.append(
          tf.where(coin,
                   tf.tile(tf.expand_dims(flip_transform, 0), [batch_size, 1]),
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if rotate > 0:
      angle_rad = rotate / 180 * math.pi
      angles = tf.random_uniform([batch_size], -angle_rad, angle_rad)
      transforms.append(
          tf.contrib.image.angles_to_projective_transforms(
              angles, height, width))

    if translate > 0:
        tx   = tf.random_uniform([batch_size,1],minval=-32,maxval=32,dtype=tf.int32)
        ty   = tf.random_uniform([batch_size,1],minval=-32,maxval=32,dtype=tf.int32)
        zero = tf.zeros([batch_size,1],dtype=tf.int32)
        one  = tf.ones([batch_size,1],dtype=tf.int32)
        ti = tf.cast(tf.concat([one,zero,tx,zero,one,ty,zero,zero],axis=1),dtype=tf.float32)
        transforms.append(ti)


    if crop_probability > 0:
      crop_pct = tf.random_uniform([batch_size], crop_min_percent,
                                   crop_max_percent)
      left = tf.random_uniform([batch_size], 0, width * (1 - crop_pct))
      top = tf.random_uniform([batch_size], 0, height * (1 - crop_pct))
      crop_transform = tf.stack([
          crop_pct,
          tf.zeros([batch_size]), top,
          tf.zeros([batch_size]), crop_pct, left,
          tf.zeros([batch_size]),
          tf.zeros([batch_size])
      ], 1)

      coin = tf.less(
          tf.random_uniform([batch_size], 0, 1.0), crop_probability)
      transforms.append(
          tf.where(coin, crop_transform,
                   tf.tile(tf.expand_dims(identity, 0), [batch_size, 1])))

    if transforms:
      #print(images.shape)
      images = tf.contrib.image.transform(
          images,
          tf.contrib.image.compose_transforms(*transforms),
          interpolation='BILINEAR') # or 'NEAREST'

    def cshift(values): # Circular shift in batch dimension
      return tf.concat([values[-1:, ...], values[:-1, ...]], 0)

    if mixup > 0:
      mixup = 1.0 * mixup # Convert to float, as tf.distributions.Beta requires floats.
      beta = tf.distributions.Beta(mixup, mixup)
      lam = beta.sample(batch_size)
      ll = tf.expand_dims(tf.expand_dims(tf.expand_dims(lam, -1), -1), -1)
      images = ll * images + (1 - ll) * cshift(images)
      labels = lam * labels + (1 - lam) * cshift(labels)

  return images#, labels

class augmentor():


    def __init__(self, nofviews = 48, ngpus = 1):
        #GPU Augmentation Graph
        # Creates a graph.
        c = []
        g_1 = tf.Graph()

        #Get number of gpus:
        nofGPUs=str(subprocess.check_output(["nvidia-smi", "-L"])).count('UUID')
        print("Number of GPUs: ",nofGPUs)
        with g_1.as_default():
        #with tf.Graph().as_default():
            with tf.device('/cpu:0'):
                            # placeholders for graph input
                self.view_ = tf.placeholder('float32', shape=(None, nofviews, 3, 224, 224), name='im0')
                view_2 = tf.reshape(self.view_, [nofGPUs,-1 , 3, 224, 224], name='first_Reshape')
                chunks =[]
                for i in range(nofGPUs):
                    chunks.append(view_2[i,:])
            for i in range(nofGPUs):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % ('tower', i)) as scope:
                    # graph outputs
                        view = tf.transpose(chunks[i], perm=[0, 2, 3, 1])
                        view = augmentImages(view, 
                            horizontal_flip=True, vertical_flip=False, translate = 64, rotate=45, crop_probability=1, mixup=0)
                        c.append(tf.transpose(view, perm=[0, 3, 1 ,2]))


            with tf.device('/cpu:0'):
              aug_view = tf.concat(c,axis=0)
              self.aug_view = tf.reshape(aug_view, shape=(-1, nofviews, 3, 224, 224))

        # Creates a session with log_device_placement set to True.
        self.sess = tf.Session(graph=g_1,config=tf.ConfigProto(log_device_placement=True,gpu_options=tf.GPUOptions(allow_growth = True)))        




    #per_process_gpu_memory_fraction=.1
    ##########
    def augment_on_GPU(self,views):
        #list_of_augviews = []
        #print("Shape of views is:",views.shape)
            val_feed_dict = {self.view_: views}
            aug_views = self.sess.run(self.aug_view, feed_dict=val_feed_dict)
        #list_of_augviews.append(aug_views)
            #print("Shape of aug views is:",aug_views.shape)

        #inputs = np.stack(list_of_augviews, axis=1)
            return aug_views
