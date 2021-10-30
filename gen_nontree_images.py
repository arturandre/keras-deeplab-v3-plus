import numpy as np
from PIL import Image
import imageio
from model import Deeplabv3
import os

from tqdm import tqdm
import tensorflow as tf

"""
The block below restricts the use to just some of the GPUs
"""
##############################################################################
# gpus = tf.config.list_physical_devices('GPU')
# if gpus:
#   # Restrict TensorFlow to only use the first GPU
#   try:
#     tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
#     #logical_gpus = tf.config.experimental.list_logical_devices('GPU')
#     #print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
#     #tf.config.experimental.set_memory_growth(logical_gpus[0], True)
#     for i in range(len(gpus)):
#         tf.config.experimental.set_memory_growth(gpus[i], True)
#   except RuntimeError as e:
#     # Visible devices must be set before GPUs have been initialized
#     print(e)
#     print('error!!!')
##############################################################################

"""
The block below allows the use of RAM memory together with
GPU memory, but it requires all the available GPUs.

Ref: https://stackoverflow.com/a/59798962/3562468
"""
##############################################################################
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 4
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
##############################################################################


# Warning - 10k+ may be dangerous!
images_per_batch = 100

imgs_folder = "/scratch/arturao/10k_nontrees/"
output_labels_folder = "/scratch/arturao/10k_nontrees_mask/"
trained_image_width=640 
mean_subtraction_value=127.5

os.makedirs(output_labels_folder, exist_ok=True)

imgs_files = os.listdir(imgs_folder)


img_filenames = []
print("Looking for image files")
for img_file in tqdm(imgs_files):
    if img_file.endswith(".png"):
        img_filenames.append(img_file)

print("Loading model")
deeplab_model = Deeplabv3(weights='cityscapes', classes=19, input_shape=(640,640,3))

processed_images = 0

while processed_images < len(img_filenames):
  print("Loading batch of found images.")
  print(f"{processed_images} processed images out of {len(img_filenames)}.")
  imgs_list = []
  for img_filename in tqdm(img_filenames[processed_images:processed_images+images_per_batch]):
    aux = os.path.join(imgs_folder, img_filename)
    img = imageio.imread(aux)
    # Normalization step
    # Ref: https://github.com/bonlime/keras-deeplab-v3-plus#how-to-get-labels
    img = (img/mean_subtraction_value)-1 
    imgs_list.append(img)

  imgs_list = np.array(imgs_list)


  #res = deeplab_model.predict(np.expand_dims(resized_image, 0))
  res = deeplab_model.predict(imgs_list, batch_size=1)
  labels = np.argmax(res, axis=-1)

  for i, label in enumerate(tqdm(labels)):
      mask_out = os.path.join(output_labels_folder, img_filenames[i+processed_images])
      np.save(mask_out, label)

  processed_images += len(imgs_list)
