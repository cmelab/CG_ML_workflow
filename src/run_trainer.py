from trainer import MLTrainer
import argparse

def get_parameters():
    parser = argparse.ArgumentParser()

    # input & output data params
    parser.add_argument('-log_dir',
                        default='/home/marjan/Documents/ml_logs',
                        type=str)
    parser.add_argument('-data_path', default="/home/marjan/Documents/dataset/pps_synthesized/dataset")
    parser.add_argument('-project', default='Synthesize_March21', type=str, help="w&b project name")
    parser.add_argument('-group', default='test', type=str, help="w&b run group")
    parser.add_argument('-notes', default='', type=str, help="w&b run note")
    parser.add_argument('-tags', default='', type=str, help="w&b run note")
    parser.add_argument('-inp_mode', default='append', type=str, help="input data mode: append or stack")

    # ML hyper params
    parser.add_argument('-lr', default=0.1, type=float, help="learning rate")
    parser.add_argument('-decay', default=0.001, type=float, help="weight decay (L2 reg)")
    parser.add_argument('-batch_size', default=16, type=int, help="batch size")
    parser.add_argument('-hidden_dim', default=10, type=int, help="number of hidden dims")
    parser.add_argument('-n_layer', default=5, type=int, help="number of layers")
    parser.add_argument('-optim', default="SGD", type=str, help="optimizer")
    parser.add_argument('-act_fn', default="Tanh", type=str, help="activation func")
    parser.add_argument('-dropout', default=0.5)

    # Run params
    parser.add_argument('-epochs', default=1000, type=int, help="number of epochs")

    args = parser.parse_args()
    return args



if __name__ == '__main__':
    args = get_parameters()
    trainer = MLTrainer(args, 123)
    trainer.run()
    #
    # from model import NN
    #
    # model = NN(in_dim=7, hidden_dim=8, energy_out_dim=1, torque_out_dim=3,
    #            n_layers=2, act_fn="Tanh", mode="append")
    # print(model)