from parser import TrainParser
from training_functions import get_model, prepare_network, train
from checkpoint_handler import save_model

def main():
    parser = TrainParser()
    args = parser.parse()

    model = get_model(args.arch)

    if not model:
        return

    prepare_network(model, args.hidden_units)
    
    train(model, args.data_directory, args.epochs, args.gpu, args.learning_rate)
    
    save_model(model, args.save_dir, args.arch)
    
if __name__ == "__main__":
    main()