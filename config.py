# epoch of batch størrelse, og lr:
# https://github.com/state-spaces/mamba/issues/8

# d_model, heads, d_ff, også flere andre parametere, antar at 16-18 lag
# https://catalog.ngc.nvidia.com/orgs/nvidia/teams/dle/resources/transformerxl_pyt

def get_config():
    return {
        "batch_size": 8, # 128 ble for stort, bruker 32 istedenfor
        "num_epochs": 20,
        "lr": 10**-4, # 10**-4 -> oversett, 1.5e-3 -> wiki 
        "seq": 350, # 1600 for wiki
        "d_model": 512,
        "d_ff": 2048,
        "h": 8, # 8
        "dropout_rate": 0.1,
        
        "stacks": 6,
                
        "weights_name": "params_11e3.pkl",
        "train_output_name": "training_log_11e3.txt",

        "datasource": 'opus_books',
        "lang_src": "en",
        "lang_tgt": "it",
        "model_folder": "weights",
        "model_basename": "tmodel_",
        "preload": "latest",
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel"
    }

