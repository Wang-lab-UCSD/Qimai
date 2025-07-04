# sei_extended.py
import torch
import torch.nn as nn
from sei_v0.train_sei import SeiProteinInteraction # Assuming original is here

class SeiProteinInteractionWithMeanEmbedding(SeiProteinInteraction):
    """
    Extends the SeiProteinInteraction model to optionally accept pre-computed
    protein embeddings (e.g., mean embedding for unknown proteins) in its
    forward pass, in addition to protein IDs.
    """
    def __init__(self, sequence_length, num_proteins, protein_emb_dim, n_genomic_features_output=1):
        # Initialize the parent class (which sets up all layers including self.protein_embedding)
        super().__init__(sequence_length, num_proteins, protein_emb_dim, n_genomic_features_output)

    def forward(self, dna_input, protein_input, protein_input_type="id"):
        """
        Forward pass for the SEI model.

        Args:
            dna_input (torch.Tensor): One-hot encoded DNA sequence tensor.
                                      Shape: (batch_size, 4, sequence_length)
            protein_input (torch.Tensor):
                If protein_input_type is "id": Protein ID tensor.
                                                Shape: (batch_size,) or (batch_size, 1)
                If protein_input_type is "embedding": Pre-computed protein embedding tensor.
                                                       Shape: (batch_size, protein_emb_dim)
            protein_input_type (str): Specifies how to interpret protein_input.
                                      Options: "id", "embedding". Defaults to "id".

        Returns:
            torch.Tensor: Logits for the interaction prediction.
                          Shape: (batch_size, n_genomic_features_output)
        """

        # --- DNA Processing (Sei Backbone) ---
        # This part is identical to the parent's forward method.
        lout1 = self.lconv1(dna_input)
        out1 = self.conv1(lout1)
        
        lout2 = self.lconv2(out1 + lout1)
        out2 = self.conv2(lout2)

        lout3 = self.lconv3(out2 + lout2)
        out3 = self.conv3(lout3)

        dconv_out1 = self.dconv1(out3 + lout3)
        cat_out1 = out3 + dconv_out1
        dconv_out2 = self.dconv2(cat_out1)
        cat_out2 = cat_out1 + dconv_out2
        dconv_out3 = self.dconv3(cat_out2)
        cat_out3 = cat_out2 + dconv_out3
        dconv_out4 = self.dconv4(cat_out3)
        cat_out4 = cat_out3 + dconv_out4
        dconv_out5 = self.dconv5(cat_out4)
        dna_features_from_cnn = cat_out4 + dconv_out5 # Shape: (batch, 960, reduced_seq_len)
        
        # Apply B-spline transformation
        # self.spline_tr_module and self._spline_df are inherited from parent
        spline_transformed_dna_features = self.spline_tr_module(dna_features_from_cnn) # Shape: (batch, 960, _spline_df)
        
        # Flatten DNA features
        dna_features_flat = spline_transformed_dna_features.view(
            spline_transformed_dna_features.size(0), 960 * self._spline_df
        )

        # --- Protein Processing ---
        protein_features_final = None
        if protein_input_type == "id":
            # protein_input is expected to be a LongTensor of shape (batch_size,) or (batch_size, 1)
            # self.protein_embedding is the nn.Embedding layer from the parent class
            protein_features_final = self.protein_embedding(protein_input)
            # Ensure shape is (batch_size, protein_emb_dim)
            if protein_features_final.ndim == 3 and protein_features_final.shape[1] == 1:
                protein_features_final = protein_features_final.squeeze(1)
        elif protein_input_type == "embedding":
            # protein_input is expected to be a FloatTensor of shape (batch_size, protein_emb_dim)
            protein_features_final = protein_input
        else:
            raise ValueError(f"Unsupported protein_input_type: '{protein_input_type}'. Must be 'id' or 'embedding'.")
        
        # Ensure protein_features_final has the correct shape for concatenation, especially if batch_size is 1
        if protein_features_final.ndim == 1 and protein_features_final.size(0) == self.protein_embedding.embedding_dim:
            # This handles the case where batch_size is 1 and a single embedding vector (emb_dim,) was passed.
            protein_features_final = protein_features_final.unsqueeze(0) # Reshape to (1, protein_emb_dim)

        # --- Combine Features & Classify ---
        if dna_features_flat.shape[0] != protein_features_final.shape[0]:
            raise ValueError(
                f"Batch size mismatch between DNA features ({dna_features_flat.shape[0]}) "
                f"and protein features ({protein_features_final.shape[0]})"
            )

        combined_features = torch.cat((dna_features_flat, protein_features_final), dim=1)
        
        # self.classifier is the nn.Sequential linear classifier from the parent class
        output_logits = self.classifier(combined_features)
        
        return output_logits