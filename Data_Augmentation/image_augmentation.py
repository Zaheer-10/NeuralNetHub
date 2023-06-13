
# Import the necessary modules
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Create an image data generator with data augmentation
datagen = ImageDataGenerator(
  rotation_range=40,
  width_shift_range=0.2,
  height_shift_range=0.2,
  shear_range=0.2,
  zoom_range=0.2,
  horizontal_flip=True,
  fill_mode="nearest"
)

# Load an image from the dataset
image = tf.keras.preprocessing.image.load_img("/content/cat.jpeg")
x = tf.keras.preprocessing.image.img_to_array(image)
x = x.reshape((1,) + x.shape)

# Generate and save augmented images
i = 0
for batch in datagen.flow(x, batch_size=1, save_to_dir="augmented", save_prefix="cat", save_format="jpeg"):
  i += 1
  if i > 20:
    break