import numpy as np
import keras.utils
from keras.datasets import mnist, fashion_mnist

class ClutteredMNISTGenerator(keras.utils.Sequence):
    def __init__(self, n_samples=10000, img_size=128, batch_size=32):
        (self.mnist_x_train, self.mnist_y_train), (self.mnist_x_test, self.mnist_y_test) = mnist.load_data()
        (self.fashion_x_train, _), (self.fashion_x_test, _) = fashion_mnist.load_data()

        self.img_size = img_size
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.num_batches = self.n_samples // self.batch_size  # Define total steps

    def __len__(self):
        return self.num_batches
    
    def embed_digits_with_clutter(self, images, labels, img_size, clutter_set):

        large_img = np.zeros((img_size, img_size), dtype=np.uint8)
        placed_boxes = []
        digit_sum = 0
        
        # Place at most 5 non-overlapping digits
        for i in range(5):
            placed = False
            attempts = 0
            while not placed and attempts < 20:
                x_offset = np.random.randint(0, img_size - 28)
                y_offset = np.random.randint(0, img_size - 28)
                digit_box = (x_offset, y_offset, x_offset + 28, y_offset + 28)
                
                if all(not (digit_box[0] < b[2] and digit_box[2] > b[0] and 
                            digit_box[1] < b[3] and digit_box[3] > b[1]) for b in placed_boxes):
                    large_img[y_offset:y_offset+28, x_offset:x_offset+28] = images[i]
                    digit_sum += labels[i]
                    placed_boxes.append(digit_box)
                    placed = True
                attempts += 1
        
        # Add at most 10 random Fashion-MNIST items as clutter
        for _ in range(10):
            placed = False
            attempts = 0
            while not placed and attempts < 10:
                clutter = clutter_set[np.random.randint(0, len(clutter_set))]
                cx_offset = np.random.randint(0, img_size - 28)
                cy_offset = np.random.randint(0, img_size - 28)
                clutter_box = (cx_offset, cy_offset, cx_offset + 28, cy_offset + 28)
                
                if all(not (clutter_box[0] < b[2] and clutter_box[2] > b[0] and 
                            clutter_box[1] < b[3] and clutter_box[3] > b[1]) for b in placed_boxes):
                    large_img[cy_offset:cy_offset+28, cx_offset:cx_offset+28] = clutter
                    placed_boxes.append(clutter_box)
                    placed = True
                attempts += 1
                
        return large_img, digit_sum

    def __getitem__(self, idx):
        
        batch_images = np.zeros((self.batch_size, self.img_size, self.img_size, 1), dtype=np.float32)
        batch_labels = np.zeros((self.batch_size,), dtype=np.int32)

        for i in range(self.batch_size):
            # Select random digits and clutter set
            digits_indices = np.random.choice(len(self.mnist_x_train), 5, replace=False)
            clutter_set = self.fashion_x_train

            # Generate cluttered MNIST image
            img, label_sum = self.embed_digits_with_clutter(
                self.mnist_x_train[digits_indices], self.mnist_y_train[digits_indices], self.img_size, clutter_set
            )

            # Normalize image and add noise
            img = img / 255.0
            noise = np.random.rand(self.img_size, self.img_size) * 0.2  # 20% noise
            img = (img * 0.8) + noise
            batch_images[i] = np.expand_dims(img, axis=-1)  # Add channel dimension
            batch_labels[i] = label_sum

        return batch_images, batch_labels
