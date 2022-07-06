# Import PyTorch
import torch
import torch.nn as nn
# Import Huggingface
# T5
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig, BertModel
from model.custom_transformer.latent_module import Latent_module

class custom_BERT(nn.Module):
    def __init__(self, isPreTrain, d_latent, variational_mode, decoder_full_model, device):
        super().__init__()

        """

        Args:

        Returns:

        """
        self.d_latent = d_latent
        self.isPreTrain = isPreTrain
        self.device = device

        if self.isPreTrain:
            self.model1 = BertModel.from_pretrained('bert-base-cased')
        else:
            model_config = BertConfig('bert-base-cased')
            self.model1 = BertModel(config=model_config)

        self.embedding = self.model1.embeddings
        self.encoder = self.model1.encoder
        self.pooler = self.model1.pooler

    def forward(self, src_input_ids, src_attention_mask,
                trg_input_ids, trg_attention_mask,
                non_pad_position=None, tgt_subsqeunt_mask=None):

        # Encoder1 Forward
        encoder_out = self.encoder1_embedding(src_input_ids)
        new_attention_mask = self.model1.get_extended_attention_mask(src_attention_mask, 
                                                                     src_attention_mask.shape, self.device)
        for i in range(len(self.encoder1_model)):
            encoder_out, _ = self.encoder1_model[i](hidden_states=encoder_out, 
                                                    attention_mask=new_attention_mask)

        encoder_out = self.encoder1_final_layer_norm(encoder_out)
        encoder_out = self.encoder1_dropout(encoder_out)

        # Latent
        if self.variational_mode != 0:
            # Target sentence latent mapping
            with torch.no_grad():
                encoder_out_trg = self.encoder1_embedding(trg_input_ids)
                new_attention_mask2 = self.model1.get_extended_attention_mask(trg_attention_mask, 
                                                                             trg_attention_mask.shape, self.device)
                for i in range(len(self.encoder1_model)):
                    encoder_out_trg, _ = self.encoder1_model[i](hidden_states=encoder_out_trg, 
                                                                attention_mask=new_attention_mask2)

            encoder_out, dist_loss = self.latent_module(encoder_out, encoder_out_trg)
        else:
            dist_loss = 0

        # Encoder2 Forward
        if self.decoder_full_model:
            model_out = self.model2(inputs_embeds=encoder_out,
                                    attention_mask=src_attention_mask,
                                    decoder_input_ids=trg_input_ids,
                                    decoder_attention_mask=trg_attention_mask)
            model_out = model_out['logits']
        else:
            model_out, _ = self.decoder_model(encoder_out)
            model_out = self.lm_head(model_out)

        return model_out, dist_loss