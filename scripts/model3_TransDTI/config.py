def BIN_config_DBPE():
    config = {}
    config['train_epoch'] = 100
    config['learning_rate'] = 0.001
    config['accumulation_steps'] = 1

    config['dropout_rate'] = 0.5
    config['batch_size'] = 64
    config['input_dim_dna'] = 768
    config['input_dim_protein'] = 384
    config['emb_dim_dna'] = 512
    config['emb_dim_protein'] = 256
    return config