def BIN_config_DBPE():
    config = {}
    config['batch_size'] = 64
    config['grad_accumul_steps'] = 1
    config['input_dim_dna'] = 768
    config['input_dim_protein'] = 384
    config['train_epoch'] = 300
    config['max_dna_seq'] = 19
    config['max_protein_seq'] = 512
    config['emb_size'] = 384
    config['dropout_rate'] = 0.2
    config['learning_rate'] = 5e-5
    
    # Encoder
    config['intermediate_size'] = 1536
    config['num_attention_heads'] = 12
    config['num_layers'] = 2
    config['attention_probs_dropout_prob'] = 0.1
    config['hidden_dropout_prob'] = 0.1
    return config