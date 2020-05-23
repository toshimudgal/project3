{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Let's construct LeNet in Keras!\n",
    "\n",
    "#### First let's load and prep our MNIST data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using TensorFlow backend.\n"
     ]
    }
   ],
   "source": [
    "from keras.preprocessing.image import ImageDataGenerator\n",
    "from keras.models import Sequential\n",
    "from keras.layers import Dense, Dropout, Activation, Flatten\n",
    "from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D\n",
    "from keras.layers.normalization import BatchNormalization\n",
    "from keras.regularizers import l2\n",
    "from keras.datasets import mnist\n",
    "from keras.utils import np_utils\n",
    "import keras\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# loads the MNIST dataset\n",
    "(x_train, y_train), (x_test, y_test)  = mnist.load_data()\n",
    "\n",
    "# Lets store the number of rows and columns\n",
    "img_rows = x_train[0].shape[0]\n",
    "img_cols = x_train[1].shape[0]\n",
    "\n",
    "# Getting our date in the right 'shape' needed for Keras\n",
    "# We need to add a 4th dimenion to our date thereby changing our\n",
    "# Our original image shape of (60000,28,28) to (60000,28,28,1)\n",
    "x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)\n",
    "x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)\n",
    "\n",
    "# store the shape of a single image \n",
    "input_shape = (img_rows, img_cols, 1)\n",
    "\n",
    "# change our image type to float32 data type\n",
    "x_train = x_train.astype('float32')\n",
    "x_test = x_test.astype('float32')\n",
    "\n",
    "# Normalize our data by changing the range from (0 to 255) to (0 to 1)\n",
    "x_train /= 255\n",
    "x_test /= 255\n",
    "\n",
    "# Now we one hot encode outputs\n",
    "y_train = np_utils.to_categorical(y_train)\n",
    "y_test = np_utils.to_categorical(y_test)\n",
    "\n",
    "num_classes = y_test.shape[1]\n",
    "num_pixels = x_train.shape[1] * x_train.shape[2]"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now let's create our layers to replicate LeNet"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "# create model\n",
    "model = Sequential()\n",
    "\n",
    "# 2 sets of CRP (Convolution, RELU, Pooling)\n",
    "#model.add(Conv2D(20, (5, 5),\n",
    " #                padding = \"same\", \n",
    " #               input_shape = input_shape))\n",
    "#model.add(Activation(\"relu\"))\n",
    "#model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))\n",
    "\n",
    "model.add(Conv2D(50, (5, 5),\n",
    "                 padding = \"same\"))\n",
    "model.add(Activation(\"relu\"))\n",
    "model.add(MaxPooling2D(pool_size = (2, 2), strides = (2, 2)))\n",
    "\n",
    "# Fully connected layers (w/ RELU)\n",
    "model.add(Flatten())\n",
    "#model.add(Dense(500))\n",
    "#model.add(Activation(\"relu\"))\n",
    "\n",
    "# Softmax (for classification)\n",
    "model.add(Dense(num_classes))\n",
    "model.add(Activation(\"softmax\"))\n",
    "           \n",
    "model.compile(loss = 'categorical_crossentropy',\n",
    "              optimizer = keras.optimizers.Adadelta(),\n",
    "              metrics = ['accuracy'])\n",
    "    \n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Now let us train LeNet on our MNIST Dataset"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Train on 60000 samples, validate on 10000 samples\n",
      "Epoch 1/3\n",
      "60000/60000 [==============================] - 53s 880us/step - loss: 0.2447 - accuracy: 0.9285 - val_loss: 0.0905 - val_accuracy: 0.9730\n",
      "Epoch 2/3\n",
      "60000/60000 [==============================] - 56s 936us/step - loss: 0.0810 - accuracy: 0.9771 - val_loss: 0.0613 - val_accuracy: 0.9803\n",
      "Epoch 3/3\n",
      "60000/60000 [==============================] - 53s 876us/step - loss: 0.0599 - accuracy: 0.9825 - val_loss: 0.0507 - val_accuracy: 0.9837\n",
      "10000/10000 [==============================] - 3s 293us/step\n",
      "Test loss: 0.05066755409669131\n",
      "Test accuracy: 0.9836999773979187\n"
     ]
    }
   ],
   "source": [
    "# Training Parameters\n",
    "batch_size = 128\n",
    "epochs = 3\n",
    "\n",
    "history = model.fit(x_train, y_train,\n",
    "          batch_size=batch_size,\n",
    "          epochs=epochs,\n",
    "          validation_data=(x_test, y_test),\n",
    "          shuffle=True)\n",
    "\n",
    "model.save(\"mnist_LeNet.h5\")\n",
    "\n",
    "# Evaluate the performance of our trained model\n",
    "scores = model.evaluate(x_test, y_test, verbose=1)\n",
    "print('Test loss:', scores[0])\n",
    "print('Test accuracy:', scores[1])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
