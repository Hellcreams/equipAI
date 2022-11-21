import tensorflow as tf

# 데이터셋 준비하기
# 1. 이미지의 크기를 설정한다.
batch_size = 128
no_classes = 10
epochs = 2
image_height, image_width = 28, 28

# 2. 케라스 유틸리티를 이용해 데이터를 디스크에서 메모리로 로드한다.
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# 3. 주어진 코드를 이용해 벡터를 이미지 형식으로 변경하고 컨볼루션을 위한 입력 크기를 정의한다.
x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)
x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)
input_shape = (image_height, image_width, 1)

# 4. 데이터 타입을 float로 변환한다.
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# 5. 데이터를 정규화한다.
x_train /= 255
x_test /= 255

# 6. 카테고리 레이블을 원-핫 인코딩으로 변환한다.
y_train = tf.keras.utils.to_categorical(y_train, no_classes)
y_test = tf.keras.utils.to_categorical(y_test, no_classes)


# 모델 구축하기
# (1)
def simple_cnn(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(
        filters=64,
        kernel_size=(3, 3),
        activation="relu"
    ))
    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(units=1024, activation="relu"))
    model.add(tf.keras.layers.Dropout(rate=0.3))
    model.add(tf.keras.layers.Dense(units=no_classes, activation="softmax"))
    model.compile(loss=tf.keras.losses.categorical_crossentropy,
                  optimizer=tf.keras.optimizers.Adam(),
                  metrics=['accuracy'])
    return model


simple_cnn_model = simple_cnn(input_shape)

# (2)
simple_cnn_model.fit(x_train, y_train, batch_size, epochs, (x_test, y_test))
train_loss, train_accuracy = simple_cnn_model.evaluate(x_train, y_train, verbose=0)
print("Train data loss:", train_loss)
print("Train data accuracy:", train_accuracy)

# (3)
test_loss, test_accuracy = simple_cnn_model.evaluate(x_test, y_test, verbose=0)
print("Test data loss:", test_loss)
print("Test data accuracy:", test_accuracy)
