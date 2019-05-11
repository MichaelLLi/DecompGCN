import argparse

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

        
class Config():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.parser.add_argument('--num_nodes', type=int,
                                 default=100, help='Number of nodes in the graph')
        self.parser.add_argument('--num_features',type=int,
                                 default=1, help='Dimension of the feature space, used in data.x')
        self.parser.add_argument('--hidden_units', type=int,
                                 default=32, help='number of units per hidden layer in the GNN')
        self.parser.add_argument('--n_layers', type=int,
                                 default=2, help='Number of layers in the GNN')
        self.parser.add_argument('--dropout_p', type=float,
                                 default=0.2, help='Dropout percentage in GNN layers')

        self.parser.add_argument('--dataset_path', type=str,
                                 default='../data/neighbor100', help='the directory to read the Dataset from')
        self.parser.add_argument('--temp_dir', type=str,
                                 default='../temp', help='directory to save temporary outputs')
        self.parser.add_argument('--summary_dir', type=str,
                                 default='summary', help='relative directory to save temporary summary')
        self.parser.add_argument('--no_summary', action='store_true',
                                 default=False, help='if passed, tensorboardx will not be used to monitor the training')
        self.parser.add_argument('--model_dir', type=str,
                                 default='model', help='relative directory to save temporary model, for both training and inference')


        self.parser.add_argument('--model', type=str,
                                 default='GCN', help='GIN | GCN | SGConv | SGIN | GAT | SGConv_Modified')
        self.parser.add_argument('--training_epochs', type=int,
                                 default=1000, help='number of training epochs')
        self.parser.add_argument('--validation_split', type=float,
                                 default=0.1, help='define size of validation set, 0 <= ... <= 1')
        self.parser.add_argument('--test_split', type=float,
                                 default=0.1, help='define size of test set, 0 <= ... <= 1')
        self.parser.add_argument('--batch_size_train', type=int,
                                 default=32, help='batch size for training')
        self.parser.add_argument('--batch_size_eval', type=int,
                                 default=32, help='batch size for evaluation')
        self.parser.add_argument('--task', type=str,
                                 default='cora', help='clique | connectivity | reddit-b | tree_cycle | triangle | planar')
        self.parser.add_argument('--residual', type=str2bool,
                                      default=True, help='residual connections')
        self.parser.add_argument('--lr', type=float,
                                      default=0.0001, help='learning rate')
        self.parser.add_argument('--lrd', type=int,
                                      default=100, help='learning rate decay patience')
        self.parser.add_argument('--layertype', type=str,
                                      default="V1,V2", help='type of connections you want to use')
        self.parser.add_argument('--normalize', type=str2bool,
                                      default=False, help='If normalize')
        self.parser.add_argument('--node_feature', type=str2bool,
                                        default=False, help='If use feature')

    def parse_args(self):
        return self.parser.parse_args()


