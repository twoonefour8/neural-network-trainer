import os
import cv2
import random
import pickle
import numpy as np
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator
from pyimagesearch.smallvggnet import SmallVGGNet


dataset = './symbols'

# инициализируем данные и метки
print("[INFO] loading images...")
data = []
labels = []

# берём пути к изображениям и рандомно перемешиваем
imagePaths = sorted(list(paths.list_images(dataset)))
random.seed(42)
random.shuffle(imagePaths)

# цикл по изображениям
for imagePath in imagePaths:
    # загружаем изображение, меняем размер на 64x64 пикселей (требуемые размеры для SmallVGGNet), изменённое изображение добавляем в список
    image = cv2.imread(imagePath)
    image = cv2.resize(image, (64, 64))
    data.append(image)

    # извлекаем метку класса из пути к изображению и обновляем список меток
    label = imagePath.split(os.path.sep)[-2]
    labels.append(label)

# масштабируем интенсивности пикселей в диапазон [0, 1]
data = np.array(data, dtype='float') / 255.0
labels = np.array(labels)

# разбиваем данные на обучающую и тестовую выборки (75% данных для обучения, 25% - для тестирования)
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.25, random_state=42)

# конвертируем метки из целых чисел в векторы
# (для 2х классов при бинарной классификации следует использовать функцию Keras 'to_categorical' вместо 'LabelBinarizer'
# из scikit-learn, которая не возвращает вектор)
lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.transform(testY)

# создаём генератор для добавления изображений
aug = ImageDataGenerator(rotation_range=30,
                         width_shift_range=0.1,
                         height_shift_range=0.1,
                         shear_range=0.2,
                         zoom_range=0.2,
                         horizontal_flip=True,
                         fill_mode="nearest"
                         )

# инициализируем VGG-подобную сверточную нейросеть
model = SmallVGGNet.build(width=64, height=64, depth=3, classes=len(lb.classes_))

# инициализируем скорость обучения, общее число эпох и размер пакета
INIT_LR = 0.01
EPOCHS = 300
BS = 32

# компилируем модель с помощью SGD (для бинарной классификации следует использовать binary_crossentropy)
print('[INFO] training network...')
opt = SGD(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

# обучаем нейросеть
H = model.fit_generator(aug.flow(trainX, trainY, batch_size=BS),
                        validation_data=(testX, testY),
                        steps_per_epoch=len(trainX) // BS,
                        epochs=EPOCHS)

# оцениваем нейросеть
print('[INFO] evaluating network...')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1), predictions.argmax(axis=1), target_names=lb.classes_))

# сохраняем модель и бинаризатор меток на диск
print('[INFO] serializing network and label binarizer...')

if not os.path.exists('model'):
    os.mkdir('model')

model.save('model/network.model')
f = open('model/network.pickle', 'wb')
f.write(pickle.dumps(lb))
f.close()
