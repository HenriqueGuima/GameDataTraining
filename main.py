# semantic segmentation with Unet
# heavily inspired on https://github.com/zhixuhao/unet
# uses isbi dataset

import os
import matplotlib.pyplot as plt
import skimage.io as skimage_io
import skimage.transform as skimage_transform
import random as r
import numpy as np
import datetime
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
import shutil

print("Tensorflow {}".format(tf.__version__))
print("GPU devices: {}".format(tf.config.list_physical_devices('GPU')))

# import PIL.Image as PImage

datasetPath = "dataset"
trainFolder = "train"
valFolder = "val"
testFolder = "test"
modelsPath = "models"

trainSize = -1  # -1 for all
valSize = -1  # -1 for all
testSize = -1  # -1 for all
exampleSize = (512, 512)
inputSize = (256, 256)
maskSize = (256, 256)
batchSize = 4
epochs = 20  # 100
learning_rate = 1e-4
numClasses = 3
showImages = False

epochs = int(epochs)
batchSize = int(batchSize)

os.makedirs(modelsPath, exist_ok=True)

# SPLIT DATASET 
def split(datasetPath, train_ratio=0.6, val_ratio=0.2):
    imagesPath = os.path.join(datasetPath, "images")
    masksPath = os.path.join(datasetPath, "masks")

    allImages = os.listdir(imagesPath)
    r.shuffle(allImages)

    totalImages = len(allImages)
    trainCount = int(totalImages * train_ratio)
    valCount = int(totalImages * val_ratio)

    trainImages = allImages[:trainCount]
    valImages = allImages[trainCount:trainCount + valCount]
    testImages = allImages[trainCount + valCount:]

    def copyFiles(files, dest):
        os.makedirs(dest, exist_ok=True)
        for file in files:
            shutil.copy(os.path.join(imagesPath, file), os.path.join(dest, file))
            shutil.copy(os.path.join(masksPath, file), os.path.join(dest, file))

    copyFiles(trainImages, os.path.join(datasetPath, trainFolder))
    copyFiles(valImages, os.path.join(datasetPath, valFolder))
    copyFiles(testImages, os.path.join(datasetPath, testFolder))

split(datasetPath, train_ratio=0.6, val_ratio=0.2)

# DATA augATION
def randomCrop(image, mask, size):
    # assert image.shape == mask.shape
    assert image.shape[0] >= size[0]
    assert image.shape[1] >= size[1]

    x = r.randint(0, image.shape[1] - size[1]+1)
    y = r.randint(0, image.shape[0] - size[0]+1)

    return image[y:y + size[0], x:x + size[1]], mask[y:y + size[0], x:x + size[1]]

def randomFlip(image, mask):
    if r.random() > 0.5:
        image = np.fliplr(image)
        mask = np.fliplr(mask)
    return image, mask

# def randomRotate(image, mask):
#     angle = r.choice([0, 90, 180, 270])
#     image = skimage_transform.rotate(image, angle)
#     mask = skimage_transform.rotate(mask, angle)
#     return image, mask

# DATA GENERATOR
class dataGenerator(tf.keras.utils.Sequence):

    def __init__(self, imagesPath, masksPath, batchSize, imageSize, aug=None):
        self.imagesPath = imagesPath
        self.masksPath = masksPath
        self.batchSize = batchSize
        self.imageSize = imageSize
        self.aug = aug
    
    def __len__(self):
        return int(np.ceil(len(self.imagesPath) / self.batchSize))
    
    def __getitem__(self, index):
        batchImages = self.imagesPath[index * self.batchSize:(index + 1) * self.batchSize]
        batchMasks = self.masksPath[index * self.batchSize:(index + 1) * self.batchSize]

        images = []
        masks = []

        for img, mask in zip(batchImages, batchMasks):
            image = skimage_io.imread(img)
            mask = skimage_io.imread(mask)

            if self.aug:
                image, mask = randomCrop(image, mask, self.imageSize)
                image, mask = randomFlip(image, mask)

            image = skimage_transform.resize(image, self.imageSize)
            mask = skimage_transform.resize(mask, self.imageSize)

            images.append(image)
            masks.append(mask)

        return np.array(images), np.array(masks)
    

# MODEL
def unet_model(input_size=(256, 256, 3), num_classes=2):
    inputs = tf.keras.layers.Input(input_size)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(inputs)
    conv1 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv1)
    pool1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(pool1)
    conv2 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv2)
    pool2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(pool2)
    conv3 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv3)
    pool3 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(pool3)
    conv4 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv4)
    pool4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(pool4)
    conv5 = tf.keras.layers.Conv2D(1024, 3, activation='relu', padding='same')(conv5)

    up6 = tf.keras.layers.Conv2D(512, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv5))
    merge6 = tf.keras.layers.concatenate([conv4, up6], axis=3)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(merge6)
    conv6 = tf.keras.layers.Conv2D(512, 3, activation='relu', padding='same')(conv6)

    up7 = tf.keras.layers.Conv2D(256, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv6))
    merge7 = tf.keras.layers.concatenate([conv3, up7], axis=3)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(merge7)
    conv7 = tf.keras.layers.Conv2D(256, 3, activation='relu', padding='same')(conv7)

    up8 = tf.keras.layers.Conv2D(128, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv7))
    merge8 = tf.keras.layers.concatenate([conv2, up8], axis=3)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(merge8)
    conv8 = tf.keras.layers.Conv2D(128, 3, activation='relu', padding='same')(conv8)

    up9 = tf.keras.layers.Conv2D(64, 2, activation='relu', padding='same')(tf.keras.layers.UpSampling2D(size=(2, 2))(conv8))
    merge9 = tf.keras.layers.concatenate([conv1, up9], axis=3)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(merge9)
    conv9 = tf.keras.layers.Conv2D(64, 3, activation='relu', padding='same')(conv9)
    conv9 = tf.keras.layers.Conv2D(num_classes, 1, activation='softmax', padding='same')(conv9)

    model = tf.keras.Model(inputs=inputs, outputs=conv9)
    return model

# CUSTOM CALLBACK FOR VISUALIZATION
class imageCallback(Callback):
    def __init__(self, val_gen, log_dir, num_images=3):
        super().__init__()
        self.val_gen = val_gen
        self.log_dir = log_dir
        self.num_images = num_images
        self.file_writer = tf.summary.create_file_writer(log_dir)

    def on_epoch_end(self, epoch, logs=None):
        val_images, val_masks = next(iter(self.val_gen))
        predictions = self.model.predict(val_images)

        with self.file_writer.as_default():
            for i in range(self.num_images):
                tf.summary.image("image", val_images[i:i+1], step=epoch)
                tf.summary.image("mask", val_masks[i:i+1], step=epoch)
                tf.summary.image("prediction", predictions[i:i+1], step=epoch)

def main():

    # CALLBACKS 
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    # modelFileName = "unet_membrane_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp + ".hdf5"
    modelFileName = "unet_membrane_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp + ".keras"
    # resultsPath = "membrane/test/predict/predict" + "E" + str(epochs) + "_LR" + str(learning_rate)
    # logs_folder = "unet_membrane_" + "E" + str(epochs) + "_LR" + str(learning_rate) + "_" + timestamp
    checkpoint_callback = ModelCheckpoint(modelFileName, monitor='val_loss', save_best_only=True)
    tensorboard_callback = TensorBoard(log_dir=f"logs/{timestamp}")
    early_stopping_callback = EarlyStopping(monitor='val_loss', patience=10)
    # image_callback = ImageCallback(val_generator=val_generator, log_dir=f"logs/{timestamp}")

    # COMPILE MODEL
    model = unet_model(input_size=(256, 256, 3), num_classes=numClasses)
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='categorical_crossentropy', metrics=['accuracy'])

    # PREPARE DATA GENERATORS
    train_images = [os.path.join(datasetPath, trainFolder, fname) for fname in os.listdir(os.path.join(datasetPath, trainFolder)) if fname.endswith('.png') or fname.endswith('.jpg')]
    val_images = [os.path.join(datasetPath, valFolder, fname) for fname in os.listdir(os.path.join(datasetPath, valFolder)) if fname.endswith('.png') or fname.endswith('.jpg')]
    test_images = [os.path.join(datasetPath, testFolder, fname) for fname in os.listdir(os.path.join(datasetPath, testFolder)) if fname.endswith('.png') or fname.endswith('.jpg')]

    train_masks = [path.replace("images", "masks") for path in train_images]
    val_masks = [path.replace("images", "masks") for path in val_images]
    test_masks = [path.replace("images", "masks") for path in test_images]

    train_generator = dataGenerator(train_images, train_masks, batchSize=batchSize, imageSize=(256, 256), aug=True)
    val_generator = dataGenerator(val_images, val_masks, batchSize=batchSize, imageSize=(256, 256), aug=False)
    test_generator = dataGenerator(test_images, test_masks, batchSize=batchSize, imageSize=(256, 256), aug=False)

    # TRAIN MODEL
    history = model.fit(train_generator, epochs=epochs, validation_data=val_generator, callbacks=[checkpoint_callback, tensorboard_callback, early_stopping_callback])

    # EVALUATE MODEL
    predictions = model.predict(test_generator)

    # CALCULATE EVALUATION METRICS
    true_labels = np.argmax(np.vstack(y for _, y in test_generator), axis=-1).flatten()

    pred_labels = np.argmax(predictions, axis=-1).flatten()

    confusion_matrix = tf.math.confusion_matrix(true_labels, pred_labels, num_classes=numClasses).numpy()
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')
    accuracy = accuracy_score(true_labels, pred_labels)

    print(f'Confusion Matrix: \n{confusion_matrix}')
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1: {f1}")
    print(f"Accuracy: {accuracy}")

    # PLOT TRAINING HISTORY

    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # SAVE MODEL
    model.save(modelFileName)
    model.save_weights(modelFileName + ".h5")

if __name__ == '__main__':
    main()
