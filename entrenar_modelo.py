import tensorflow as tf
from tensorflow.keras import layers, models

def entrenar_modelo():
    print("Descargando y preparando el dataset MNIST...")
    # Cargar la base de datos MNIST
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalizar los valores de los píxeles (de 0-255 a 0.0-1.0)
    x_train, x_test = x_train / 255.0, x_test / 255.0

    # Expandir dimensiones para que tengan el canal de color (28, 28, 1)
    x_train = x_train[..., tf.newaxis]
    x_test = x_test[..., tf.newaxis]

    print("Construyendo la Red Neuronal Convolucional...")
    # Crear el modelo CNN
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax') # 10 clases de salida (dígitos del 0 al 9)
    ])

    # Compilar el modelo
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    print("Iniciando el entrenamiento...")
    # Entrenar el modelo
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    # Guardar el modelo en disco
    nombre_archivo = 'modelo_mnist.h5'
    model.save(nombre_archivo)
    print(f"\n¡Entrenamiento finalizado! Modelo guardado exitosamente como '{nombre_archivo}'")

if __name__ == "__main__":
    entrenar_modelo()