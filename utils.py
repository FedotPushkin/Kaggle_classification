def show_images(random_image, random_image_path,random_image_cat):
    plt.figure(figsize=(15,8))
    for index,path in enumerate(random_image_path):
        #im = PIL.Image.open(f'{PATH}train/{path}')
        im = PIL.Image.open(f'{path}')
        plt.subplot(3,3, index+1)
        plt.imshow(im)
        plt.title(f'Class: {str(random_image_cat[index])}, size: {im.size}')
        plt.axis('off')
    plt.show()

def imshow(image_RGB):
    io.imshow(image_RGB)
    io.show()

def plot_history(history):
    plt.figure(figsize=(10,5))
    #plt.style.use('dark_background')
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'b', label='Training acc')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    #plt.figure()
    plt.figure(figsize=(10,5))
    #plt.style.use('dark_background')
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()