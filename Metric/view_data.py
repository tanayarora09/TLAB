from PIL import Image
from Helper import get_cifar
import keras as K
import keras.layers as lyr #type:ignore


def show_augmented_images(dataset, data_augmentation, num_batches=3):
    num_img = 0
    for batch_num, (images, labels) in enumerate(dataset.take(num_batches)):
        augmented_images = data_augmentation(images)
        for i in range(len(augmented_images)):
            img = augmented_images[i].numpy()
            img = (img * 255).astype('uint8')  # Convert to uint8 format
            img_pil = Image.fromarray(img)
            img_pil.save(f'data_viewing/{num_img}.jpg')
            num_img += 1

dt, dv = get_cifar()

show_augmented_images(dt, K.Sequential([lyr.RandomCrop(200, 200),  lyr.RandomZoom(0.2), lyr.RandomRotation(0.1), K.layers.RandomContrast(0.3), lyr.Resizing(224, 224)]))