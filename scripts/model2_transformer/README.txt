#### training model ####
> main_old.py and models_old.py are the direct application of transformerCPI model on DPI dataset

> main.py and models.py are the first successiful transformer model on the pre-trained embeddings

> main_v2.py didn't load pre-trained embeddings directly. Instead, it searches the embeddings in the database and then loads them during training

> main_v2.py generates pre-trained embeddings on the fly. It turns out to be too slow. Training on a 24,000-sample dataset, single epoch takes 70 min. This is the old main_v2.py, somehow the script is gone

> main_v3.py trains the model with alternating sub-dataset, i.e. trains on 1st split for several epochs, and then trains on another split for several epochs.

> main_v4.py is modified based on v3. It pads protein sequence and DNA sequence to max length of each batch instead of fixed length

> main_v5.py is modified based on v3. It deals with the regression problem and uses models_v2.py

> main_v6.py is modified based on v3. For each epoch, it iterates over the whole datase, which is a more common practice and can avoid the fluctuation in learning curve

> main_v7.py is modified based on v3. It has functions of v4, i.e. pads protein sequence and DNA sequence to max length of each batch instead of fixed length

> main_v8.py is modified based on v3. It uses models_v3. It removes protein input completely to test how much protein module contributes to the prediction

> main_v9.py is modified based on v3. It applies clustered CV strategy to split train/val dataset. It aims to keep several TF families from training

> main_v10.py is modified based on v3. It uses models_CR.py as model

> main_v11.py is modified based on v10. It uses models_CR.py as model and selects dna binding regions of proteins instead of randomly selecting max_p sub-sequence.

> main_v12.py is modified based on v3. It extracts cross attention from decoder and uses it for final prediction.

> main_v13.py is modified based on v3. It uses models_v5.py.


#### evaluating model ####
> model_eval.py evaluates the model with alternating sub-dataset on all TFs together

> model_eval_v2.py evaluates the model with alternating sub-dataset and evaluates on individual TF

> model_eval_v3.py evaluates the regression model. It returns the predicted values and true values for plotting regression plot


#### model ####
> models.py is the original transformer model for classification. DNA is "target language" and protein is "source language"

> models_v2.py is the regression model. The architecture is the same as o.g. The only difference is that the loss function is MSE instead BCE

> models_v3.py is the model which only contains DNA module.

> models_CR.py is modified from models_attention.py. This model adds CR implementation, i.e first generate a weight matrix, and then get the dot product between weight and feature.

> models_CR_v2.py is modified from models_attention.py. This model adds alternative CR implementation

> models_v4.py is modified from models_attention.py. This model extracts cross attention from decoder and uses it for final prediction.

> models_v5.py is modified from models.py. This model switches the protein module and DNA module.

> models_v6.py is modified from models.py. In this model, DNA and protein modules are identical. Both DNA and protein will go through encoder and decoder module. And then take the mean of learned DNA and protein embeddings to go through prediction module


> models_attention.py is modified from models.py. This model can extract attention weights.

> models_attention_v2.py is modified from models_v5.py. This model can extract attention weights.

> models_attention_v3.py is modified from models_v6.py. This model can extract attention weights.

