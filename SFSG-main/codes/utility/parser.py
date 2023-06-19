import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument('--data_path', nargs='?', default='/home/WeiYiBiao/LA-baseline/data/',
                        help='Input data path.')
    parser.add_argument('--seed', type=int, default=2023,
                        help='Random seed')
    parser.add_argument('--dataset', nargs='?', default='sports',
                        help='Choose a dataset from {sports, baby,tiktok}')
    parser.add_argument('--verbose', type=int, default=1,
                        help='Interval of evaluation.')
    parser.add_argument('--epoch', type=int, default=1024,
                        help='Number of epoch.')
    parser.add_argument('--batch_size', type=int, default=2048,
                        help='Batch size.')
    parser.add_argument('--regs', nargs='?', default='[1e-2,1e-3,1e-2]',
                        help='Regularizations.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--model_name', nargs='?', default='SFSG',
                        help='Specify the model name.')
    parser.add_argument('--embed_size', type=int, default=64,
                        help='Embedding size.')
    parser.add_argument('--feat_embed_dim', type=int, default=128,
                        help='')                        
    parser.add_argument('--weight_size', nargs='?', default='[64,64]',
                        help='Output sizes of every layer')
    parser.add_argument('--core', type=int, default=5,
                        help='5-core for warm-start')
    parser.add_argument('--topk', type=int, default=10,
                        help='K value of k-NN sparsification')  
    parser.add_argument('--lambda_coeff', type=float, default=0.9,
                        help='Lambda value of skip connection')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of item graph conv layers')  
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1]',
                        help='Keep probability w.r.t. message dropout (i.e., 1-dropout_ratio) for each deep layer. 1: no dropout.')
    parser.add_argument('--early_stopping_patience', type=int, default=20,
                        help='') 
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='GPU id')
    parser.add_argument('--Ks', nargs='?', default='[10, 20]',
                        help='K value of ndcg/recall @ k')
    parser.add_argument('--test_flag', nargs='?', default='part',
                        help='Specify the test type from {part, full}, indicating whether the reference is done in mini-batch')
    parser.add_argument('--nbit', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.1, help='')
    parser.add_argument('--nhead', type=int, default=4, help='"nhead" in Transformer.')
    parser.add_argument('--act', type=str, default='gelu', help='"activation" in Transformer.')
    parser.add_argument('--num_layer', type=int, default=2, help='"num_layer" in Transformer.')
    parser.add_argument('--ssl_temp', type=float, default=0.05, help='温度系数')
    parser.add_argument('--ssl_reg', type=float, default=5*1e-6, help='对比学习损失函数')
    parser.add_argument('--drop_node', type=int, default=1, help='是否在节点级别构建对比学习')
    parser.add_argument('--ssl_temp1', type=float, default=0.05, help='温度系数')
    parser.add_argument('--ssl_reg1', type=float, default=1e-7, help='对比学习损失函数')    
    parser.add_argument('--contrastive_i_i', type=int, default=1, help='是否在节点级别构建对比学习')


    return parser.parse_args()
