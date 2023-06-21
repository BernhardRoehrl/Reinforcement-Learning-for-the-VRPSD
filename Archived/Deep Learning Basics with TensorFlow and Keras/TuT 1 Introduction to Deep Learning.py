import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis =1)
x_test = tf.keras.utils.normalize(x_test, axis =1)

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))  # default activation check or write in thesis why this "rectified linear"
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))  # this is the output layer with number of classifications: "10". Probability distribution -> softmax

model.compile(optimizer='adam', loss ='sparse_categorical_crossentropy', metrics=['accuracy'])  # optimizer there are like 10 this is the most complicated part and i need to check which there are
#  loss = degree of error maybe us binary here, you can change adam if you know what you do.

model.fit(x_train, y_train, epochs=3)

val_loss, val_acc = model.evaluate(x_test, y_test)  # validation its not only memorizing (to close to much of delta?)
print(val_loss, val_acc)

model.save('epic_num_reader.model')
new_model = tf.keras.models.load_model('epic_num_reader.model')
predictions = new_model.predict([x_test])  # ALWAYS TAKES A LIST
print(predictions)

print(np.argmax(predictions[0]))

plt.imshow(x_train[0], cmap=plt.cm.binary)
plt.show()

#print(x_train[0])