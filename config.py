# coding: utf-8 

class DefaultConfig(object):

    model = 'PCNN_ONE'  # the name of used model, in  <models/__init__.py>
    data = 'NYT'  # SEM NYT FilterNYT

    result_dir = './out'
    load_model_path = 'checkpoints/model.pth'  # the trained model

    seed = 3435
    batch_size = 128  # batch size
    use_gpu = True  # user GPU or not
    gpu_id = 0
    num_workers = 0  # how many workers for loading data

    max_len = 80  # max_len for each sentence + two padding

    word_dim = 50
    pos_dim = 5

    norm_emb=True

    num_epochs = 16  # the number of epochs for training
    drop_out = 0.5
    lr = 0.0003  # initial learning rate
    lr_decay = 0.95  # when val_loss increase, lr = lr*lr_decay
    weight_decay = 0.0001  # optimizer parameter

    # Conv
    filters = [3]
    filters_num = 230
    sen_feature_dim = filters_num

opt = DefaultConfig()