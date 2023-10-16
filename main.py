# Verify Reading Dataset via MnistDataloader class
#
import random
import matplotlib.pyplot as plt
from mnistReader import *
from os.path import join
from os import getcwd

#
# Set file paths based on added MNIST Datasets
#
input_path = getcwd()
training_images_filepath = join(input_path, 'dataset\\train-images.idx3-ubyte')
training_labels_filepath = join(input_path, 'dataset\\train-labels.idx1-ubyte')
test_images_filepath = join(input_path, 'dataset\\t10k-images.idx3-ubyte')
test_labels_filepath = join(input_path, 'dataset\\t10k-labels.idx1-ubyte')

def main():
    #
    # Load MINST dataset
    #
    print(join(input_path, 't10k-labels-idx1-ubyte\\train-labels-idx1-ubyte'))
    mnist_dataloader = MnistDataloader(training_images_filepath, training_labels_filepath, test_images_filepath, test_labels_filepath)
    (x_train, y_train), (x_test, y_test) = mnist_dataloader.load_data()

    #
    # Show some random training and test images 
    #
    images_2_show = []
    titles_2_show = []
    for i in range(0, 10):
        r = random.randint(1, 60000)
        images_2_show.append(x_train[r])
        titles_2_show.append('training image [' + str(r) + '] = ' + str(y_train[r]))    

    for i in range(0, 5):
        r = random.randint(1, 10000)
        images_2_show.append(x_test[r])        
        titles_2_show.append('test image [' + str(r) + '] = ' + str(y_test[r]))    

    show_images(images_2_show, titles_2_show)
    #menu()

#
# Helper function to show a list of images with their relating titles
#
def show_images(images, title_texts):
    cols = 5
    rows = int(len(images)/cols) + 1
    plt.figure(figsize=(30,20))
    index = 1    
    for x in zip(images, title_texts):        
        image = x[0]        
        title_text = x[1]
        plt.subplot(rows, cols, index)        
        plt.imshow(image, cmap=plt.cm.gray)
        if (title_text != ''):
            plt.title(title_text, fontsize = 15);        
        index += 1



def menu():
    print("Welcome to my neural networking project!")
    # Number of hidden layers (max 16)
    hidden_num = 17
    while hidden_num <1 and hidden_num>16:
        hidden_num = int(input("Please enter how many hidden layers you would like (Max 16): "))
    # Number of neurons per hidden layer (max 8)
    hidden_neurons = []
    for i in hidden_num:
        hidden_neurons.append(input(f"Please enter how many neurons you would like in hidden layer {i} (Max 8): "))
    # Desired activation function
    activation_func = input("Enter the following for your desired activation function:"
                       +"\n\t S - Sigmoid"
                       +"\n\t R - ReLU"
                       +"\n\t L - Leaky ReLU"
                       +"\n\t P - PReLU"
                       +"\n\t E - ELU"
                       +"\n\t T - Tanh"
                       +"\n Choice: ")
    #Desired learning rate
    learning_rate = input("Please enter your desired learning rate (Min - 0.00001)(Max - 10): ")


    
    





main()