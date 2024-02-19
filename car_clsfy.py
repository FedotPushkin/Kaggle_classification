import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import zipfile
import csv
import sys
import os
import dill
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler,ModelCheckpoint
from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
from tensorflow.keras.regularizers import l2
from tensorflow.keras import optimizers
import tensorflow.keras.models  as M
import tensorflow.keras.layers as L
from tensorflow.keras.applications.xception import Xception
from sklearn.model_selection import train_test_split,StratifiedKFold
import PIL
from PIL import ImageOps, ImageFilter
from pylab import rcParams
import efficientnet.tfkeras  as efn
import efficientnet
import zipfile
from skimage import io
#Код скопирован из Jupyter notebook, поэтому структуры почти нет
#to run app install efficientnet
#!pipinstall - q efficientnet
#!nvidia-smi
def main():

    rcParams['figure.figsize'] = 10, 5
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))


    print(os.listdir("../input"))
    print('Python       :', sys.version.split('\n')[0])
    print('Numpy        :', np.__version__)
    print('Tensorflow   :', tf.__version__)
    print('Keras        :', keras.__version__)
    print('effinet       :' ,efficientnet.__version__)
    #!pip freeze >requirements.txt
    EPOCHS               = 20  # эпох на обучение
    BATCH_SIZE           = 64 # уменьшаем batch если сеть большая, иначе не поместится в память на GPU
    LR                   = 1e-3
    VAL_SPLIT            = 0.2 # сколько данных выделяем на тест = 20%

    CLASS_NUM            = 10  # количество классов в нашей задаче
    IMG_SIZE             = 250 # какого размера подаем изображения в сеть
    IMG_CHANNELS         = 3   # у RGB 3 канала
    INPUT_SHAPE          = (IMG_SIZE, IMG_SIZE, IMG_CHANNELS)

    DATA_PATH = '../input/sf-dl-car-classification/'
    MY_SAVED_DATA_PATH = '../input/car-classification/'
    PATH = '../working/car/' # рабочая директория

    RANDOM_SEED = 77

    CLASS_NAMES = (
      'Приора', #0
      'Ford Focus', #1
      'Самара', #2
      'ВАЗ-2110', #3
      'Жигули', #4
      'Нива', #5
      'Калина', #6
      'ВАЗ-2109', #7
      'Volkswagen Passat', #8
      'ВАЗ-21099' #9
    )
    train_df = pd.read_csv(DATA_PATH+"train.csv")
    sample_submission = pd.read_csv(DATA_PATH + "sample-submission.csv")
    print(train_df.value_counts(sort=True))
    train_df['Category'] = train_df['Category'].apply(lambda num:str(num))
    sample_submission['Category'] = sample_submission['Category'].apply(lambda num:str(num))

    train_df['Category'].value_counts(sort=False).plot(kind = 'barh', figsize=(12, 3))

    train_df['Id'].nunique()

    print('Unzipping images...')

    for data_zip in ["train.zip","test.zip"]:
        with zipfile.ZipFile(DATA_PATH + data_zip,"r") as z:
            z.extractall(PATH)
    print('Unzipping done')

    for directory in os.listdir(PATH+'train'):
        for (dirpath, dirnames, filenames) in os.walk(f'{PATH}train/{directory}'):
            for file in filenames:
                # move file
                os.replace(f'{dirpath}/{file}', f'{PATH}train/{file}')
            # remove directory after copy
            os.rmdir(dirpath)

    count = 0
    for root_dir, cur_dir, files in os.walk(PATH + '/train'):
        count += len(files)
    print('file count:', count)



    abs_file_names = []

    for file_name in train_df['Id']:
        tmp = os.path.abspath(f'{PATH}train'+os.sep+file_name)
        abs_file_names.append(tmp)

    # update dataframe
    train_df['Id'] = abs_file_names
    train_df.head()

    plt.figure(figsize=(15,8))
    random_image= train_df.sample(n=9)
    random_image_path = random_image['Id'].values
    random_image_cat = random_image['Category'].values

    show_images(random_image, random_image_path, random_image_cat)

    train_files, test_files, train_labels, test_labels = train_test_split(train_df['Id'], train_df['Category'],test_size = 0.2,random_state = RANDOM_SEED, stratify = train_df['Category'])

    train_files = pd.DataFrame(train_files)
    test_files = pd.DataFrame(test_files)
    train_files['Category'] = train_labels
    test_files['Category'] = test_labels
    #train_files.shape, test_files.shape

    #test_files.head()
    #train_files.head()
    train_files['Category'].value_counts(sort=False).plot(kind='barh', figsize=(15, 4))

    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=50,
        shear_range=0.2,
        zoom_range=[0.75, 1.25],
        brightness_range=[0.5, 1.5],
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True
    )

    test_datagen = ImageDataGenerator(
        rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_files,
        directory=f'{PATH}train',
        x_col="Id",
        y_col="Category",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED,)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_files,
        directory=f'{PATH}train',
        x_col="Id",
        y_col="Category",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=RANDOM_SEED,)


    x,y = train_generator.next()
    print('Пример картинок из train_generator')
    plt.figure(figsize=(12,8))

    for i in range(0,6):
        image = x[i]
        plt.subplot(3,3, i+1)
        plt.imshow(image)
    plt.show()

    x,y = test_generator.next()
    print('Пример картинок из test_generator')
    plt.figure(figsize=(12,8))

    for i in range(0,6):
        image = x[i]
        plt.subplot(3,3, i+1)
        plt.imshow(image)
    plt.show()

    base_model = efn.EfficientNetB7(weights='imagenet', include_top=False, input_shape=INPUT_SHAPE)
    base_model.trainable = False
    model = M.Sequential()
    model.add(base_model)
    model.add(L.GlobalAveragePooling2D(),)
    model.add(L.Dense(CLASS_NUM, activation='softmax'))
    print(len(model.layers))
    print(len(model.trainable_variables))
    LR = 0.001
    model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=LR), metrics=["accuracy"])
    checkpoint = ModelCheckpoint('best_model.hdf5', monitor=['val_accuracy'], verbose=1, mode='max')
    earlystop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
    callbacks_list = [checkpoint, earlystop]

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_files,
        directory=f'{PATH}train',
        x_col="Id",
        y_col="Category",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED,)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_files,
        directory=f'{PATH}train',
        x_col="Id",
        y_col="Category",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False,
        seed=RANDOM_SEED,)

    scores = model.evaluate(test_generator, steps=1, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))

    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples//train_generator.batch_size,
        validation_data=test_generator,
        validation_steps=test_generator.samples//test_generator.batch_size,
        epochs=EPOCHS,
        callbacks=callbacks_list)

    model.save('../working/model_step1.hdf5')
    model.load_weights('best_model.hdf5')

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_files,
        directory=f'{PATH}train',
        x_col="Id",
        y_col="Category",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE//2,
        class_mode='categorical',
        shuffle=True,
        seed=RANDOM_SEED,)

    test_generator = test_datagen.flow_from_dataframe(
        dataframe=test_files,
        directory=f'{PATH}train',
        x_col="Id",
        y_col="Category",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE//2,
        class_mode='categorical',
        shuffle=False,
        seed=RANDOM_SEED,)

    scores = model.evaluate_generator(test_generator, verbose=1)
    print("Accuracy: %.2f%%" % (scores[1]*100))
    plot_history(history)

    print("Number of layers in the base model: ", len(base_model.layers))
    base_model.trainable = True

    # Fine-tune from this layer onwards
    fine_tune_at = 2*len(base_model.layers)//3

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable =  False
        len(base_model.trainable_variables)

        LR = 0.0001
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=LR), metrics=["accuracy"])

        scores = model.evaluate_generator(test_generator, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=test_generator,
            validation_steps=test_generator.samples // test_generator.batch_size,
            epochs=EPOCHS,
            callbacks=callbacks_list
        )

        model.save('../working/model_step2.hdf5')
        model.load_weights('best_model.hdf5')
        model.load_weights('best_model.hdf5')
        scores = model.evaluate(test_generator, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))

        plot_history(history)

        base_model.trainable = True

        # Fine-tune from this layer onwards
        fine_tune_at = len(base_model.layers) // 3

        # Freeze all the layers before the `fine_tune_at` layer
        for layer in base_model.layers[:fine_tune_at]:
            layer.trainable = False
        LR = 0.000007
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=LR), metrics=["accuracy"])

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_files,
            diretory=f'{PATH}train',
            x_col="Id",
            y_col="Category",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE // 4,
            class_mode='categorical',
            shuffle=True,
            seed=RANDOM_SEED,
            validate_filenames=True,
            # class_mode='input',
        )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_files,
            directory=f'{PATH}train',
            x_col="Id",
            y_col="Category",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE // 4,
            class_mode='categorical',
            shuffle=False,
            seed=RANDOM_SEED)

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=test_generator,
            validation_steps=test_generator.samples // test_generator.batch_size,
            epochs=20,
            callbacks=callbacks_list
        )

        model.save('../working/model_step3.hdf5')

        plot_history(history)

        base_model.trainable = True
        LR = 0.0000007
        model.compile(loss="categorical_crossentropy", optimizer=optimizers.Adam(learning_rate=LR), metrics=["accuracy"])

        train_generator = train_datagen.flow_from_dataframe(
            dataframe=train_files,
            diretory=f'{PATH}train',
            x_col="Id",
            y_col="Category",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE // 8,
            class_mode='categorical',
            shuffle=True,
            seed=RANDOM_SEED,
            validate_filenames=True,
            # class_mode='input',
            )

        test_generator = test_datagen.flow_from_dataframe(
            dataframe=test_files,
            directory=f'{PATH}train',
            x_col="Id",
            y_col="Category",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE // 8,
            class_mode='categorical',
            shuffle=False,
            seed=RANDOM_SEED, )

        history = model.fit(
            train_generator,
            steps_per_epoch=train_generator.samples // train_generator.batch_size,
            validation_data=test_generator,
            validation_steps=test_generator.samples // test_generator.batch_size,
            epochs=40,
            callbacks=callbacks_list
        )

        model.save('../working/model_step4.hdf5')
        model.load_weights('best_model.hdf5')
        scores = model.evaluate(test_generator, verbose=1)
        print("Accuracy: %.2f%%" % (scores[1] * 100))
        plot_history(history)
        test_datagen = ImageDataGenerator(rescale=1. / 255,
                                        rotation_range=90,
                                        shear_range=0.2,
                                        zoom_range=[0.75, 1.25],
                                        brightness_range=[0.5, 1.5],
                                        width_shift_range=0.1,
                                        height_shift_range=0.1, )
        test_sub_generator = test_datagen.flow_from_dataframe(
            dataframe=sample_submission,
            directory=PATH + 'test_upload/',
            x_col="Id",
            y_col=None,
            shuffle=False,
            class_mode=None,
            seed=RANDOM_SEED,
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE, )

        preds_wo_tta = model.predict_generator(test_sub_generator, steps=1, verbose=1)
        tta_steps = 10
        predictions = []

        for i in range(tta_steps):
            preds = model.predict_generator(test_sub_generator, steps=len(test_sub_generator), verbose=1)
            predictions.append(preds)

        pred = np.mean(predictions, axis=0)

        print(predictions)
        predictions = np.argmax(pred, axis=-1)  # multiple categories
        label_map = (train_generator.class_indices)
        label_map = dict((v, k) for k, v in label_map.items())  # flip k,v
        predictions = [label_map[k] for k in predictions]
        # preds_wo_tta = [label_map[k] for k in preds_wo_tta]
        filenames_with_dir = test_sub_generator.filenames
        submission = pd.DataFrame({'Id': filenames_with_dir, 'Category': predictions}, columns=['Id', 'Category'])
        submission['Id'] = submission['Id'].replace('test_upload/', '')
        submission.to_csv('submission_.csv', index=False)
        # submission2 = pd.DataFrame({'Id':filenames_with_dir, 'Category':preds_wo_tta}, columns=['Id', 'Category'])
        # submission2['Id'] = submission['Id'].replace('test_upload/','')
        # submission2.to_csv('submission_wo_tta.csv', index=False)
        print('Save submit')