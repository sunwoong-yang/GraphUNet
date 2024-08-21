import argparse


def parsing():
	# Initialize the argument parser
	parser = argparse.ArgumentParser()

	# Add arguments to the parser
	# parser.add_argument('-b', '--beta', default=0, type=float, help="Value of beta")
	parser.add_argument('-train', default='False', type=str, help="Whether to train models")
	parser.add_argument('-tr', '--trainlist', default='4', type=str, help="Index of train data")
	parser.add_argument('-norm', default='L', type=str, help="Whether to apply GraphNorm(G/g)/LayerNorm(L/l)/None")
	parser.add_argument('-te', '--testlist', default='', type=str, help="Index of test data")
	parser.add_argument('-mode', '--testmode', default='auto,100', type=str,
	                    help="Test with ROM('ROM')/auto-regressive('auto')/train-data('train')/test-data('test')")

	# parser.add_argument('-s', '--skip', default='False', type=str, help="Skip-connection (T/F)")
	parser.add_argument('-sc', '--scaler', default='1', type=int, help="Scaler type (1~4)")
	parser.add_argument('-n', '--name', default='Case_Folder_Name', type=str, help="Custom name")
	# parser.add_argument('-d', '--decoder', default=6, type=int, help="Decoder type")
	parser.add_argument('-e', '--epoch', default=4000, type=int, help="Epochs")
	parser.add_argument('-u', '--unet', default='True', type=str, help="Unet")
	parser.add_argument('-his', '--history', default=40, type=int, help="Dimension of training features")
	parser.add_argument('-lstmhis', '--lstm_history', default=40, type=int,
	                    help="Dimension of temporal model's training features")
	parser.add_argument('-f', '--future', default='True', type=str, help="Where to predict future snapshots")
	parser.add_argument('-lr', default=1e-3, type=float, help="Initial learning rate")
	parser.add_argument('-mini', '--minibatch', default=25, type=int, help="Minibatch size")
	parser.add_argument('-tr_idx', '--train_idx', default='150, 300', type=str, help="Snapshot range of train data")
	parser.add_argument('-EHC', '--Enc_HC', default='40,20,10,5,1', type=str, help="Hidden channels of the Encoder")
	parser.add_argument('-DHC', '--Dec_HC', default='1,1,1,1,1', type=str, help="Hidden channels of the Decoder")
	# parser.add_argument('-mlp', '--mlp_layer',default='50, 25', type=str, help="MLP for latent space")
	parser.add_argument('-gcn', '--gcn_type', default='0', type=str,
	                    help=("Type of GCN: "
		      "[0: GMMConv] "
		      "[1: ChebConv] "
		      "[2: GCNConv (x improved & o edgefeature)] "
		      "[3: GATConv] "
		      "[4: GCNConv (o improved & o edgefeature)] "
		      "[5: GCNConv (x improved & x edgefeature)] "
		      "[6: GCNConv (o improved & x edgefeature)]")
	)
	parser.add_argument('-k', '--gcn_k', default=2, type=int, help="Chebyshev's filter or GMM's kernel size")
	# parser.add_argument('-p', '--pooling', default='1000, 500, 250, 100', type=str, help="Pooling layers")
	parser.add_argument('-p', '--pooling', default='0.7,0.7,0.7,0.7', type=str, help="Pooling layers")
	parser.add_argument('-z', '--zero_unpooling', default='True', type=str, help="Whether to unpool with zero values")
	parser.add_argument('-mp', '--my_pooling', default='False', type=str, help="Custom Pooling?")
	parser.add_argument('-a', '--augmentation_order', default=1, type=int,
	                    help="Order of adjacency augmentation after pooling?")
	parser.add_argument('-nt', '--noise_type', default=0, type=int,
	                    help="Noise type: 0 -> w/o/ noise, 1 -> noise in input, 2 -> noise in in/output")
	parser.add_argument('-ns', '--noise_size', default=0.02, type=float,
	                    help="Noise size")
	parser.add_argument('-r', '--n_repeat', default=1, type=int, help="Number of repeat")
	parser.add_argument('-snap', default='False', type=str, help="Whether to apply different number of snapshots used for training: for slow shedding, snapshots' range becomes 150~350")
	parser.add_argument('-trans', default='None', type=str, help="Whether transductive learning")

	# Parse the arguments
	args = parser.parse_args()

	args.train_idx = [int(item) for item in args.train_idx.split(',')]
	args.testmode = [item for item in args.testmode.split(',')]
	if not (isinstance(args.train_idx, list) and len(args.train_idx) == 2):
		raise ValueError("args.train_idx is invalid: it should be a list with 2 elements")

	Enc_HC_ = [int(item) for item in args.Enc_HC.split(',')]
	args.Enc_HC = [1, 3, 5, 3, 1] if not args.future.lower() in ['t', 'true'] else Enc_HC_
	args.Dec_HC = [int(item) for item in args.Dec_HC.split(',')]

	if args.gcn_type.lower() in ['0', 'gmm']:  # GMMConv
		args.gcn_type = 0
	elif args.gcn_type.lower() in ['1', 'chebconv', 'spectral']:  # ChebConv
		args.gcn_type = 1
	elif args.gcn_type.lower() in ['2', 'gcn', 'vanilla']:  # GCNConv
		args.gcn_type = 2
	elif args.gcn_type.lower() in ['3', 'gat']:  # GCNConv
		args.gcn_type = 3
	elif args.gcn_type.lower() in ['4', 'gcn_improved', 'vanilla_improved']:  # GCNConv_improved (A+2I)
		args.gcn_type = 4
	elif args.gcn_type.lower() in ['5']:
		args.gcn_type = 5
	elif args.gcn_type.lower() in ['6']:
		args.gcn_type = 6
	else:
		raise ValueError("Invalid GCN type: check 'args.gcn_type'")

	if_pool = args.pooling.lower() not in ['f', 'false']

	def parse_item(item):
		try:
			return int(item)
		except ValueError:
			return float(item)

	args.pooling = [parse_item(item) for item in args.pooling.split(',')] if if_pool else None

	args.future = args.future.lower() in ['t', 'true']
	args.unet = args.unet.lower() in ['t', 'true']
	args.zero_unpooling = args.zero_unpooling.lower() in ['t', 'true']
	args.my_pooling = args.my_pooling.lower() in ['t', 'true']
	args.snap = args.snap.lower() in ['t', 'true']
	args.trans = None if args.trans.lower() in ['none'] else parse_item(args.trans)
	# args.augmentation = args.augmentation.lower() in ['t', 'true']

	return args
