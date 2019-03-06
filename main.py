from hed import *
from keras.utils import plot_model
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger
from keras import backend as K
import os
from data_parser import DataParser
import cv2


def generator(dataParser, train_data=True):
    while True:
        if train_data:
            batch_ids = np.random.choice(dataParser.train_ids, dataParser.batch_size)
        else:
            batch_ids = np.random.choice(dataParser.validate_ids, dataParser.batch_size * 2)
        images, edge_maps, _ = dataParser.get_batch_data(batch_ids)
        yield (images, [edge_maps, edge_maps, edge_maps, edge_maps, edge_maps,
                        edge_maps])  # 5 side output and 1 fuse output


def train(model, dataParser, epoches):
    model_name = 'HED-Keras'
    model_dir = os.path.join('./checkpoints', model_name)
    csv_log = os.path.join(model_dir, 'train_log.csv')
    checkpoint = os.path.join(model_dir, 'checkpoint.{epoch:02d}-{val_loss:.2f}.hdf5')

    checkpointer = ModelCheckpoint(filepath=checkpoint, verbose=1, save_best_only=True)  # save model after each epoch
    csv_logger = CSVLogger(filename=csv_log, separator=';', append=True)
    tensorboard = TensorBoard(log_dir=model_dir, histogram_freq=1, batch_size=10, write_graph=False, write_grads=True,
                              write_images=False)

    train_history = model.fit_generator(generator(dataParser),
                                        steps_per_epoch=dataParser.step_per_epoch,
                                        epochs=epoches,
                                        verbose=1,
                                        callbacks=[checkpointer, csv_logger, tensorboard],
                                        validation_data=generator(dataParser, False),
                                        validation_steps=dataParser.validation_steps)
    return train_history


def generate_edge(model, input_imgs, names, output_dir):
    prediction = model.predict(input_imgs)
    # ...


if __name__ == "__main__":
    # K.set_image_data_format(data_format='channels_last')
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0'

    model = hed()
    plot_model(model, to_file=os.path.join('./model', 'model.png'), show_shapes=True)

    dataParser = DataParser(batch_size=10)

    # training
    train_history = train(model, dataParser, 200)

    # generating
    # input_dir = ''
    # output_dir = ''
    # input_images = os.listdir(input_dir)
    # input = []
    # names = []
    # for img in input_images:
    #     input_img = cv2.imread(img)
    #     input.append(input_img)
    #     names.append(img.split('/.')[-2])
    # generate_edge(model, input, names, output_dir)


