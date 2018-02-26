from bigbrain.tf_network import SiameseNetwork

def build_net(input_var=None, input_shape=[(None, 1, 1019, 1019),(None, 1, 1019, 1019)]):
    siam = SiameseNetwork(input_shape[0], summary=False)
    siam.add_conv_layer(num_filters=16, filter_size=5, stride=2, batch_norm=False, summary=False)
    siam.add_block(num_filters=16, width=2)
    siam.add_block(num_filters=32, width=2)
    siam.add_block(num_filters=32, width=2)
    siam.add_block(num_filters=64, width=2)
    siam.add_block(num_filters=64, width=2)
    siam.add_block(num_filters=128, width=2)
    siam.add_dense_layer(num_units=500)
    siam.combine_branches(num_labels=1, distance_function='L1', weights='vector')
    return siam

