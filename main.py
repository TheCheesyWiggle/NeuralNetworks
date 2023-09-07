def main():
    menu()

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
    hidden_num = input("Please enter your desired learning rate (Min - 0.00001)(Max - 10): ")
    
    





main()