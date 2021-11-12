import tensorflow as tf
from keras.models import Sequential, Model, Input
from keras.layers import LSTM, Dense, Embedding
from keras.layers.merge import concatenate

class RNNModel:
    def __init__(self):
        self.nl_seq_dim = (265, 100)
        # self.nl_seq = Sequential()
        # self.nl_seq.add(LSTM(output_dim=384, input_shape=(82201,384)))
        # self.nl_seq.dense()

    def define_model(self):
        nl_seq_input = Input(shape=self.nl_seq_dim, name='nl_input')
        nl_seq_output = Embedding()(nl_seq_input)
        nl_model = Model(nl_seq_input, nl_seq_output)

        up_seq_input = Input(shape=(47,13), name='up_input')
        up_seq_lstm = LSTM(265, dropout=0.2, input_shape=(265,13))(up_seq_input)
        up_seq_output = Dense(5, name='up_output')(up_seq_lstm)
        up_seq_model = Model(up_seq_input, up_seq_output)

        concat = concatenate([nl_seq_output, up_seq_output])
        model_output = Dense(13, activation='softmax')(concat)

        final_model = Model([nl_seq_input, up_seq_input], model_output)
        final_model.compile(loss='categorical_crossentropy',optimizer='adam', metrics=['accuracy'])
        print(final_model.summary())
        tf.keras.utils.plot_model(final_model, to_file="plot.png", show_shapes=True)
        plt.show()
        return final_model