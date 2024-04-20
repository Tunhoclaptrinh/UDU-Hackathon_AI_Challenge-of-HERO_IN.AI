import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras import datasets, layers, models
(training_images, training_labels), (test_images, test_labels) = datasets.cifar10.load_data()
training_images = training_images / 255.0
test_images = test_images / 255.0

class_names = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']

for i in range(16):
    plt.subplot(4, 4, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])

plt.show()
train_images = training_images[:20000]
train_labels = training_labels[:20000]
test_images = test_images[:4000]
test_labels = test_labels[:4000]

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

loss, accuracy = model.evaluate(test_images, test_labels, verbose=2)
print('Loss: ', loss)
print('Test accuracy:', accuracy)

model.save('imagenet_classifier')
model = models.load_model('imagenet_classifier')

img = cv2.imread('asd asdsadas đá')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img, cmap=plt.cm.binary)

prediction = model.predict(np.array([img])/255)
print(prediction)