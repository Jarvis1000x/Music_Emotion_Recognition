import tensorflow.keras as keras
import tensorflow_addons as tfa


class MultiOutputModel:

    def __init__(self, input_shape):

        self.input_shape = input_shape
        self.base_model = None
        self.A2EBranch = None
        self.A2Mid2EBranch = None
        self.A2Mid2EJointBranch = None

        self._build()
        self.model = None

    def _build(self):
        self._build_base_model()
        self._build_A2E()
        self._build_A2Mid2E()
        self._build_A2Mid2EJointBranch()
        self._build_total_model()

    def _build_base_model(self, inputs):

        inputs = keras.layers.Conv2D(64, (5, 5), strides=2, activation="relu", padding="valid")(inputs)
        x = keras.layers.BatchNormalization()(inputs)

        # 2nd Layer
        x = keras.layers.Conv2D(64, (3, 3), strides=1, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # 3rd Layer
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.3)(x)

        # 4th Layer
        x = keras.layers.Conv2D(128, (3, 3), strides=1, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # 5th Layer
        x = keras.layers.Conv2D(128, (3, 3), strides=1, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # 6th Layer
        x = keras.layers.MaxPooling2D((2, 2))(x)
        x = keras.layers.Dropout(0.3)(x)

        # 7th Layer
        x = keras.layers.Conv2D(256, (3, 3), strides=1, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # 8th Layer
        x = keras.layers.Conv2D(256, (3, 3), strides=1, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # 9th Layer
        x = keras.layers.Conv2D(384, (3, 3), strides=1, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # 10th Layer
        x = keras.layers.Conv2D(512, (3, 3), strides=1, activation="relu", padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        # 11th Layer
        x = keras.layers.Conv2D(256, (3, 3), strides=1, activation="relu", padding="same")(x)
        common_layer = keras.layers.BatchNormalization()(x)

        # 12th Layer
        # x = tfa.layers.AdaptiveAveragePooling2D(x)

        return common_layer

    def _model_input(self):
        return keras.Input(shape=self.input_shape, name="Model Input")

    def _build_A2E(self, input):

        first = self._build_base_model(input)
        branch = keras.layers.Dense(256)(first)

        A2E_branch = keras.layers.Dense(8, activation="softmax")(branch)

        return A2E_branch

    def _build_A2Mid2E(self, input):

        first = self._build_base_model(input)
        branch = keras.layers.Flatten()(first)
        branch = keras.layers.Dense(256)(branch)

        branch = keras.layers.Dense(7)(branch)
        branch = keras.layers.Dense(7, activation="softmax")(branch)

        branch = keras.layers.Dense(8)(branch)
        A2Mid2E_branch = keras.layers.Dense(8, activation="softmax")(branch)

        return A2Mid2E_branch

    def _build_A2Mid2EJointBranch(self, input):

        first = self._build_base_model(input)
        branch = keras.layers.Flatten()(first)
        branch = keras.layers.Dense(256)(branch)

        branch = keras.layers.Dense(7)(branch)

        branch = keras.layers.Dense(8)(branch)
        A2Mid2EJoint_branch = keras.layers.Dense(8, activation="softmax")(branch)

        return A2Mid2EJoint_branch

    def _build_total_model(self):

        model_input = self._model_input()
        inputs = self._build_base_model(model_input)
        A2E_B = self._build_A2E(inputs)
        A2Mid2E_B = self._build_A2Mid2E(inputs)
        A2Mid2EJoint_B = self._build_A2Mid2EJointBranch(inputs)
        self.model = keras.Model(inputs=inputs,
                                 outputs=[A2E_B, A2Mid2E_B, A2Mid2EJoint_B],
                                 name="Emotion Detection")

    def compile(self, learning_rate=0.0001):

        optimizer = keras.optimizers.Adam(learning_rate=learning_rate)
        self.model.compile(optimizer=optimizer,
                           loss={
                               'A2E_branch': 'categorical_crossentropy',
                               'A2Mid2E_branch': 'categorical_crossentropy',
                               'A2Mid2EJoint_branch': 'mse'},
                           )

    def train(self, x_train, y_train, batch_size, num_epochs):

        self.model.fit(x_train,
                       y_train,
                       batch_size=batch_size,
                       epochs=num_epochs,
                       shuffle=True)


