import math
# Import PyTorch
import torch
import torch.nn as nn
from torch.autograd import Variable
# Import Huggingface
# T5
from transformers import T5ForConditionalGeneration, T5EncoderModel, T5Config, T5TokenizerFast
# Bart
from transformers import BartTokenizerFast, BartForConditionalGeneration, BartConfig
#
from ..loss import GaussianKLLoss

class Pretrained_Transformer(nn.Module):
    def __init__(self, model_type, isPreTrain, variational_mode, d_latent):
        super().__init__()

        """
        Initialize WAE model
        
        Args:
            encoder_config (dictionary): encoder transformer's configuration
            d_latent (int): latent dimension size
            device (torch.device): 
        Returns:
            log_prob (torch.Tensor): log probability of each word 
            mean (torch.Tensor): mean of latent vector
            log_var (torch.Tensor): log variance of latent vector
            z (torch.Tensor): sampled latent vector
        """
        self.d_latent = d_latent
        self.model_type = model_type
        self.isPreTrain = isPreTrain

        if self.model_type == 'T5':
            if self.isPreTrain:
                self.model = T5ForConditionalGeneration.from_pretrained('t5-small')
            else:
                model_config = T5Config.from_pretrained('t5-small')
                self.model = T5ForConditionalGeneration(config=model_config)

            # Encoder1 Setting
            self.encoder1_embedding = self.model1.encoder.embed_tokens
            self.encoder1_model = self.model1.encoder.block
            self.encoder1_final_layer_norm = self.model1.encoder.final_layer_norm
            self.encoder1_dropout = self.model1.encoder.dropout
            # Dimension Setting
            self.d_hidden = self.encoder1_embedding.embedding_dim
            # Encoder2 Setting
            self.encoder2_model = self.model2.get_encoder()
            # Decoder Setting
            self.decoder_model = self.model2.get_decoder()
            # Final Layer Setting
            self.vocab_size = self.model2.lm_head.out_features
            self.lm_head = self.model2.lm_head

        elif self.model_type == 'bart':
            if self.isPreTrain:
                self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
            else:
                model_config = BartConfig.from_pretrained("facebook/bart-base")
                self.model = BartForConditionalGeneration(config=model_config)
                self.model.init_weights()

            # Shared embedding setting
            self.embeddings = self.model.model.shared
            # Encoder Setting
            self.encoder_model = self.model.get_encoder()
            # Dimension Setting
            self.d_hidden = self.encoder_model.embed_tokens.embedding_dim
            # Decoder Setting
            self.decoder_model = self.model.get_decoder()
            # Final Layer Setting
            self.vocab_size = self.model.lm_head.out_features
            self.lm_head = self.model.lm_head

        # Variational model setting
        self.variational_mode = variational_mode
        if self.variational_mode == 1:
            self.context_to_mu = nn.Linear(self.d_hidden, d_latent)
            self.context_to_logvar = nn.Linear(self.d_hidden, d_latent)
            self.mu_to_context = nn.Linear(d_latent, self.d_hidden)
            self.logvar_to_context = nn.Linear(d_latent, self.d_hidden)

            self.kl_criterion = GaussianKLLoss()
            self.latent_to_decoder = nn.Linear(self.d_hidden*2, self.d_hidden)


    def forward(self, src_input_ids, src_attention_mask, trg_input_ids, trg_attention_mask, 
                non_pad_position=None, tgt_subsqeunt_mask=None):

    #===================================#
    #===============Bart================#
    #===================================#

        if self.model_type == 'bart':

            # # Input and Output embedding
            # src_input_embeds = self.embeddings(src_input_ids)
            # trg_input_embeds = self.embeddings(trg_input_ids)

            # # Encoder Forward
            # src_encoder_out = self.encoder_model(inputs_embeds=src_input_embeds,
            #                                      attention_mask=src_attention_mask)
            # src_encoder_out = src_encoder_out['last_hidden_state']

            # if self.variational_mode == 1:
            #     # Source sentence latent mapping
            #     src_mu = self.context_to_mu(src_encoder_out)
            #     src_logvar = self.context_to_logvar(src_encoder_out)
            #     # Target sentence latent mapping
            #     with torch.no_grad():
            #         trg_encoder_out = self.encoder_model(input_ids=trg_input_ids,
            #                                         attention_mask=trg_attention_mask)
            #         trg_encoder_out = trg_encoder_out['last_hidden_state']

            #     trg_mu = self.context_to_mu(trg_encoder_out)
            #     trg_logvar = self.context_to_logvar(trg_encoder_out)

            #     kl = self.kl_criterion(src_mu, src_logvar, trg_mu, trg_logvar)

            #     mu = self.mu_to_context(src_mu)
            #     logvar = self.logvar_to_context(src_logvar)

            #     std = logvar.mul(0.5).exp_()
            #     eps = Variable(std.data.new(std.size()).normal_())
            #     z = eps.mul(std).add_(mu)

            #     src_encoder_out = torch.cat([src_encoder_out, z], dim=2)
            #     src_encoder_out = self.latent_to_decoder(src_encoder_out)
            # else:
            #     kl = 0

            # # Decoder
            # model_out = self.decoder_model(inputs_embeds=trg_input_embeds, 
            #                                attention_mask=trg_attention_mask,
            #                                encoder_hidden_states=src_encoder_out,
            #                                encoder_attention_mask=src_attention_mask)
            # model_out = self.lm_head(model_out['last_hidden_state'])

            # if non_pad_position is not None:
            #     model_out = model_out[non_pad_position]

            # testing
            model_out = self.model(input_ids=src_input_ids, attention_mask=src_attention_mask)
            model_out = model_out.logits

            kl = 0

            if non_pad_position is not None:
                model_out = model_out[non_pad_position]

            return model_out, kl

    #===================================#
    #=============== T5 ================#
    #===================================#

        # elif self.model_type == 'T5':
        #     # Encoder1 Forward
        #     wae_enc_out = self.encoder1_embedding(input_ids)
        #     new_attention_mask = self.model1.get_extended_attention_mask(attention_mask, 
        #                                                                  attention_mask.shape, self.device)
        #     for i in range(len(self.encoder1_model)):
        #         wae_enc_out, _ = self.encoder1_model[i](hidden_states=wae_enc_out, 
        #                                                 attention_mask=new_attention_mask)

        #     wae_enc_out = self.encoder1_final_layer_norm(wae_enc_out)
        #     wae_enc_out = self.encoder1_dropout(wae_enc_out)

        #     # Encoder2 Forward
        #     wae_dec_out = self.encoder2_model(inputs_embeds=wae_enc_out, 
        #                                       attention_mask=attention_mask)
        #     wae_dec_out = wae_dec_out['last_hidden_state']

        #     # Decoder
        #     if self.decoder_type == 'Transformer':
        #         model_out = self.decoder_model(inputs_embeds=wae_enc_out, 
        #                                     attention_mask=attention_mask,
        #                                     encoder_hidden_states=wae_dec_out,
        #                                     encoder_attention_mask=attention_mask)
        #         model_out = self.lm_head(model_out['last_hidden_state'])
        #     else:
        #         model_out, _ = self.decoder_model(wae_enc_out)
        #         model_out = self.decoder_linear1(model_out)
        #         model_out = self.decoder_linear2(model_out)

        #     return wae_enc_out, wae_dec_out, model_out