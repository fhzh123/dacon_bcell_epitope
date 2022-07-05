# Import PyTorch
import torch
from torch import nn
from torch.autograd import Variable
# Import custom modules
from .encoder_decoder import full_cnn_latent_encoder, full_cnn_latent_decoder, cnn_latent_encoder, cnn_latent_decoder
from .loss import GaussianKLLoss, MaximumMeanDiscrepancyLoss

class Latent_module(nn.Module):
    def __init__(self, d_model, d_latent, variational_mode, z_var, token_length, device):

        super(Latent_module, self).__init__()

        self.variational_mode = variational_mode
        self.z_var = z_var
        self.loss_lambda = 1
        self.device = device
        
        # Variational Autoencoder
        if self.variational_mode in [1,2]:
            self.context_to_mu = nn.Linear(d_model, d_latent)
            self.context_to_logvar = nn.Linear(d_model, d_latent)
            self.z_to_context = nn.Linear(d_latent, d_model)

        # Wasserstein Autoencoder
        if self.variational_mode in [3,4]:
            self.context_to_latent = nn.Linear(d_model, d_latent)
            self.latent_to_context = nn.Linear(d_latent, d_model)

            self.mmd_criterion = MaximumMeanDiscrepancyLoss(device=self.device)

        # CNN + Variational Autoencoder
        if self.variational_mode in [5,6]:
            self.context_to_mu = nn.Linear(d_latent, d_latent)
            self.context_to_logvar = nn.Linear(d_latent, d_latent)

            if self.variational_mode == 5:
                self.latent_encoder = full_cnn_latent_encoder(d_model, d_latent, token_length)
                self.latent_decoder = full_cnn_latent_decoder(d_model, d_latent, token_length)
            if self.variational_mode == 6:
                self.latent_encoder = cnn_latent_encoder(d_model, d_latent)
                self.latent_decoder = cnn_latent_decoder(d_model, d_latent)

        # CNN + Wasserstein Autoencoder
        if self.variational_mode in [7,8]:

            self.mmd_criterion = MaximumMeanDiscrepancyLoss(device=self.device)

            if self.variational_mode == 7:
                self.latent_encoder = full_cnn_latent_encoder(d_model, d_latent, token_length)
                self.latent_decoder = full_cnn_latent_decoder(d_model, d_latent, token_length)
            if self.variational_mode == 7:
                self.latent_encoder = cnn_latent_encoder(d_model, d_latent)
                self.latent_decoder = cnn_latent_decoder(d_model, d_latent)

        # Gaussian Mixture Variational Autoencoder
        if self.variational_mode == 9:

            self.context_to_cls = nn.Linear(d_model, 2)
            self.context_to_mu = nn.Linear(d_model + 2, d_latent)
            self.context_to_logvar = nn.Linear(d_model + 2, d_latent)
            
            self.z_to_context = nn.Linear(d_latent, d_model)
        
        # Style & Semantic Divied
        if self.variational_mode == 9:
            self.content_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.style_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.content_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )
            self.style_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )

            # Define Gaussian Mixture
            mix = torch.distributions.Categorical(torch.ones(2))
            mean_1 = torch.zeros(d_latent) - 2 # (d_latent)
            mean_2 = torch.zeros(d_latent) + 2 # (d_latent)
            sigma_1 = torch.ones(d_latent) * 0.5 # (d_latent)
            sigma_2 = torch.ones(d_latent) * 0.5 # (d_latent)
            comp = torch.distributions.Independent(torch.distributions.Normal(torch.stack([mean_1, mean_2]), torch.stack([sigma_1, sigma_2])), 1) # (2, d_latent)
            self.style_latent_gmm = torch.distributions.MixtureSameFamily(mix, comp) # 

            self.mmd_criterion = MaximumMeanDiscrepancyLoss(device=self.device)
            self.content_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()
            self.style_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()
        
        if self.variational_mode == 10:
            self.content_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.style_latent_encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=4, stride=4), # (batch_size, d_model, 60)
                nn.GELU(),
                nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=5), # (batch_size, d_model, 10)
                nn.GELU(),
                nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=5, stride=1), # (batch_size, d_model, 1)
                nn.GELU()
            )

            self.content_latent_to_mu = nn.Linear(d_latent, d_latent)
            self.content_latent_to_logvar = nn.Linear(d_latent, d_latent)
            self.style_latent_to_mu = nn.Linear(d_latent, d_latent)
            self.style_latent_to_logvar = nn.Linear(d_latent, d_latent)

            self.content_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )
            self.style_latent_decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.GELU()
            )

            self.kl_criterion = GaussianKLLoss()
            self.content_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()
            self.style_similiarity_criterion = nn.CosineEmbeddingLoss() #nn.CosineSimilarity()

    def forward(self, encoder_out_src, encoder_out_trg=None):

    #===================================#
    #=========VAE (W. Average)==========#
    #===================================#

        """
        1. Model dimension to latent dimenseion with 'context_to_mu' [seq_len, batch, d_latent]
        2. Average sequence token [batch, d_latent]
        3. Calculate Gaussian KL-Divergence
        4. Re-parameterization trick
        5. Decoding by 'z_to_context'
        """

        if self.variational_mode == 1:
            # 1. Model dimension to latent dimenseion with 'context_to_mu'
            src_mu = self.context_to_mu(encoder_out_src) # [seq_len, batch, d_latent]
            src_logvar = self.context_to_logvar(encoder_out_src) # [seq_len, batch, d_latent]

            trg_mu = self.context_to_mu(encoder_out_trg) # [seq_len, batch, d_latent]
            trg_logvar = self.context_to_logvar(encoder_out_trg) # [seq_len, batch, d_latent]

            # 2. Average sequence token
            src_mu = src_mu.mean(dim=0) # [batch, d_latent]
            src_logvar = src_logvar.mean(dim=0) # [batch, d_latent]

            trg_mu = trg_mu.mean(dim=0) # [batch, d_latent]
            trg_logvar = trg_logvar.mean(dim=0) # [batch, d_latent]

            # 3. Calculate Gaussian KL-Divergence
            numerator = src_logvar.exp() + torch.pow(src_mu - trg_mu, 2)
            fraction = torch.div(numerator, (trg_logvar.exp()))
            dist_loss = 0.5 * torch.sum(trg_logvar - src_logvar + fraction - 1, dim=1)
            dist_loss = dist_loss.mean() # Batch mean

            # 4. Re-parameterization
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu) # [batch, d_latent]

            # 5. Decoding by 'z_to_context'
            resize_z = self.z_to_context(z) # [batch, d_model]

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #===========VAE (W. View)===========#
    #===================================#

        """
        1. Model dimension to latent dimenseion 
           with 'context_to_mu' and 'context_to_logvar' [seq_len, batch, d_latent]
        2. Reshape latent variable [batch, seq_len * d_latent]
        3. Calculate Gaussian KL-Divergence
        4. Re-parameterization trick
        5. Decoding by 'z_to_context'
        """

        if self.variational_mode == 2:
            # 1. Model dimension to latent dimenseion
            src_mu = self.context_to_mu(encoder_out_src) # [seq_len, batch, d_latent]
            src_logvar = self.context_to_logvar(encoder_out_src) # [seq_len, batch, d_latent]

            trg_mu = self.context_to_mu(encoder_out_trg) # [seq_len, batch, d_latent]
            trg_logvar = self.context_to_logvar(encoder_out_trg) # [seq_len, batch, d_latent]

            # 2. Reshape latent variable
            batch_size = encoder_out_src.size(1)
            src_mu = src_mu.view(batch_size, -1) # [batch, seq_len * d_latent]
            src_logvar = src_logvar.view(batch_size, -1) # [batch, seq_len * d_latent]

            trg_mu = trg_mu.view(batch_size, -1) # [batch, seq_len * d_latent]
            trg_logvar = trg_logvar.view(batch_size, -1) # [batch, seq_len * d_latent]

            # 3. Calculate Gaussian KL-Divergence
            numerator = src_logvar.exp() + torch.pow(src_mu - trg_mu, 2)
            fraction = torch.div(numerator, (trg_logvar.exp()))
            dist_loss = 0.5 * torch.sum(trg_logvar - src_logvar + fraction - 1, dim=1)
            dist_loss = dist_loss.mean() # Batch mean

            # 4. Re-parameterization trick
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu) # [batch, d_latent]

            # 5. Decoding with 'z_to_context'
            resize_z = self.z_to_context(z) # [batch, d_model]

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #=============WAE(mean)=============#
    #===================================#

        """
        1. Model dimension to latent dimenseion with 'context_to_latent' [seq_len, batch, d_latent]
        2. Average sequence token [batch, d_latent]
        3. Calculate Maximum-mean discrepancy
        4. Decoding with 'latent_to_context'
        """

        if self.variational_mode == 3:
            # 1. Model dimension to latent dimenseion
            src_latent = self.context_to_latent(encoder_out_src) # [seq_len, batch, d_latent]
            trg_latent = self.context_to_latent(encoder_out_trg) # [seq_len, batch, d_latent]

            # 2. Average sequence token
            src_latent = src_latent.mean(dim=0)
            trg_latent = trg_latent.mean(dim=0)

            # 3. Calculate Maximum-mean discrepancy 
            dist_loss = self.mmd_criterion(src_latent, trg_latent, self.z_var)

            # 4. Decoding with 'latent_to_context'
            src_latent = self.latent_to_context(src_latent)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #=============WAE(view)=============#
    #===================================#

        """
        1. Model dimension to latent dimenseion with 'context_to_latent' [seq_len, batch, d_latent]
        2. Reshape latent variable [batch, seq_len * d_latent]
        3. Calculate Maximum-mean discrepancy
        4. Decoding with 'latent_to_context'
        """

        if self.variational_mode == 4:
            # 1. Model dimension to latent dimenseion
            src_latent = self.context_to_latent(encoder_out_src) # [seq_len, batch, d_latent]
            trg_latent = self.context_to_latent(encoder_out_trg) # [seq_len, batch, d_latent]

            # 2. Reshape latent variable
            batch_size = encoder_out_src.size(1)
            src_latent = src_latent.view(batch_size, -1)
            trg_latent = trg_latent.view(batch_size, -1)

            # 3. Calculate Maximum-mean discrepancy 
            dist_loss = self.mmd_criterion(src_latent, trg_latent, self.z_var)

            # 4. Decoding with 'latent_to_context'
            src_latent = self.latent_to_context(src_latent)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #==============CNN+VAE==============#
    #===================================#

        """
        1. Model dimension to latent dimenseion 
           with CNN encoder and max-over time pooling [batch, d_latent]
        2. Latent variable to mu and log variance [batch, d_latent]
        3. Calculate Gaussian KL-Divergence
        4. Re-parameterization trick
        5. Decoding with CNN Decoder
        """

        if self.variational_mode in [5,6]:

            # 1. Model dimension to latent dimenseion with CNN encoder
            src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
            trg_latent = self.latent_encoder(encoder_out_trg) # [batch, d_latent]

            # 2. Latent variable to mu and log variance
            src_mu = self.context_to_mu(src_latent) # (batch, d_latent)
            src_logvar = self.context_to_logvar(src_latent) # (batch, d_latent)

            trg_mu = self.context_to_mu(trg_latent) # (batch, d_latent)
            trg_logvar = self.context_to_logvar(trg_latent) # (batch, d_latent)
            
            # 3. Calculate Gaussian KL-Divergence
            numerator = src_logvar.exp() + torch.pow(src_mu - trg_mu, 2)
            fraction = torch.div(numerator, (trg_logvar.exp()))
            dist_loss = 0.5 * torch.sum(trg_logvar - src_logvar + fraction - 1, dim=1)
            dist_loss = dist_loss.mean() # Batch mean

            # 4. Re-parameterization trick
            std = src_logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            z = eps.mul(std).add_(src_mu) # [batch, d_latent]

            # 5. Decoding with CNN Decoder
            resize_z = self.latent_decoder(z.unsqueeze(2)) # [seq_len, batch, d_model]

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #==============CNN+WAE==============#
    #===================================#

        """
        1. Model dimension to latent dimenseion 
           with CNN encoder and max-over time pooling [batch, d_latent]
        2. Calculate Maximum-mean discrepancy
        3. Decoding with CNN Decoder
        """

        if self.variational_mode in [7,8]:

            # 1. Model dimension to latent dimenseion with CNN encoder
            src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
            trg_latent = self.latent_encoder(encoder_out_trg) # [batch, d_latent]

            # 2. Calculate Maximum-mean discrepancy
            dist_loss = self.mmd_criterion(src_latent, trg_latent, self.z_var)

            # 3. Decoding with full CNN Decoder
            src_latent = self.latent_decoder(src_latent.unsqueeze(2)) # [seq_len, batch, d_latent]

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #======Gaussian Mixture + VAE=======#
    #===================================#

        """
        Need refactoring
        """

        if self.variational_mode == 9:

            # # 1. Style classifiction 
            # x_to_cls = encoder_out_src.mean(dim=0) # [batch, d_model]
            # distribution_cls = self.context_to_cls(x_to_cls) # [batch, num_cls]
            # distribution_cls_cp = distribution_cls.unsqueeze(0).repeat(encoder_out_src.size(0), 1, 1) # [token, batch, num_cls]

            # # concat
            # gm_inp = torch.cat((encoder_out_src, distribution_cls_cp), dim=2) # (tokn, batch, d_latent + num_cls)

            # src_mu = self.context_to_mu(gm_inp) # (token, batch, d_latent)
            # src_logvar = self.context_to_logvar(gm_inp) # (token, batch, d_latent)

            # mu = src_mu.view(gm_inp.size(1), -1) # [batch, seq_len * d_latent]
            # logvar = src_logvar.view(gm_inp.size(1), -1) # [batch, seq_len * d_latent]
            # dist_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # std = src_logvar.mul(0.5).exp_()
            # eps = Variable(std.data.new(std.size()).normal_())
            # z = eps.mul(std).add_(src_mu)

            # # 5. Decoding with 'z_to_context'
            # resize_z = self.z_to_context(z) # [batch, d_model]

            # encoder_out_total = torch.add(encoder_out_src, resize_z)

            raise Exception('Need refactoring...!! Try another variational mode')

    #===================================#
    #=========CNN+VAE(GMM&SIM)==========#
    #===================================#

        if self.variational_mode == 10:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)
            encoder_out_trg = encoder_out_trg.permute(1, 2, 0) # To: (batch_size, d_model, seq_len)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_content_latent = self.content_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)
            trg_content_latent = trg_content_latent.squeeze(2) # (batch_size, d_latent)

            # 1-2. VAE Process
            # 1-2-1. Get mu and logvar from src_content_latent and trg_content_latent
            src_content_mu = self.content_latent_to_mu(src_content_latent) # (batch_size, d_latent)
            src_content_logvar = self.content_latent_to_logvar(src_content_latent) # (batch_size, d_latent)
            trg_content_mu = self.content_latent_to_mu(trg_content_latent) # (batch_size, d_latent)
            trg_content_logvar = self.content_latent_to_logvar(trg_content_latent) # (batch_size, d_latent)

            # 1-2-2. Reparameterization trick
            src_content_std = src_content_logvar.mul(0.5).exp_()
            src_content_eps = torch.randn_like(src_content_std).to(self.device)
            src_content_z = src_content_eps * src_content_std + src_content_mu # (batch_size, d_latent)

            trg_content_std = trg_content_logvar.mul(0.5).exp_()
            trg_content_eps = torch.randn_like(trg_content_std).to(self.device)
            trg_content_z = trg_content_eps * trg_content_std + trg_content_mu # (batch_size, d_latent)

            # 1-2-3. Get kl loss with N~(0, 1)
            kl_loss_content_src = self.kl_criterion(src_content_mu, src_content_logvar,
                                                    torch.Tensor([0]).to(self.device),
                                                    torch.Tensor([1]).to(self.device))
            kl_loss_content_trg = self.kl_criterion(trg_content_mu, trg_content_logvar,
                                                    torch.Tensor([0]).to(self.device),
                                                    torch.Tensor([1]).to(self.device))

            # 1-3. Similarity Loss - src_content_latent and trg_content_latent
            sim_loss_content = self.content_similiarity_criterion(src_content_z, trg_content_z, torch.Tensor([1]).to(self.device))

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_style_latent = self.style_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            trg_style_latent = trg_style_latent.squeeze(2) # (batch_size, d_latent)

            # 2-2. VAE Process
            # 2-2-1. Get mu and logvar from src_style_latent and trg_style_latent
            src_style_mu = self.style_latent_to_mu(src_style_latent) # (batch_size, d_latent)
            src_style_logvar = self.style_latent_to_logvar(src_style_latent) # (batch_size, d_latent)
            trg_style_mu = self.style_latent_to_mu(trg_style_latent) # (batch_size, d_latent)
            trg_style_logvar = self.style_latent_to_logvar(trg_style_latent) # (batch_size, d_latent)

            # 2-2-2. Reparameterization trick
            src_style_std = src_style_logvar.mul(0.5).exp_()
            src_style_eps = torch.randn_like(src_style_std).to(self.device)
            src_style_z = src_style_eps * src_style_std + src_style_mu # (batch_size, d_latent)

            trg_style_std = trg_style_logvar.mul(0.5).exp_()
            trg_style_eps = torch.randn_like(trg_style_std).to(self.device)
            trg_style_z = trg_style_eps * trg_style_std + trg_style_mu # (batch_size, d_latent)

            # 2-2-3. Get kl loss with N~(0, 1)
            kl_loss_style_src = self.kl_criterion(src_style_mu, src_style_logvar, 
                                                  torch.Tensor([0]).to(self.device),
                                                  torch.Tensor([1]).to(self.device))
            kl_loss_style_trg = self.kl_criterion(trg_style_mu, trg_style_logvar,
                                                  torch.Tensor([0]).to(self.device),
                                                  torch.Tensor([1]).to(self.device))

            # 2-3. Similarity Loss - src_style_latent and trg_style_latent
            sim_loss_style = self.style_similiarity_criterion(src_style_z, trg_style_z, torch.Tensor([-1]).to(self.device))

            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_z.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_z.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 4. Define dist_loss
            dist_loss = kl_loss_content_src + kl_loss_content_trg + kl_loss_style_src + kl_loss_style_trg
            sim_loss = sim_loss_content + sim_loss_style # maximize sim_loss_content, minimize sim_loss_style
            
            dist_loss = dist_loss + sim_loss

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

    #===================================#
    #=========CNN+WAE(GMM&SIM)==========#
    #===================================#

        if self.variational_mode == 11:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)
            encoder_out_trg = encoder_out_trg.permute(1, 2, 0) # To: (batch_size, d_model, seq_len)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_content_latent = self.content_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)
            trg_content_latent = trg_content_latent.squeeze(2) # (batch_size, d_latent)

            # 1-2. WAE Process
            # 1-2-1. Draw fake content latent from N~(0,2)
            fake_content_latent = torch.randn_like(src_content_latent).to(self.device) * 2 # (batch_size, d_latent)

            # 1-2-2. get mmd_loss between src_content_latent and fake_content_latent, trg_content_latent and fake_content_latent
            mmd_loss_content_src = self.mmd_criterion(src_content_latent, fake_content_latent, 2) # z_var is 2 now
            mmd_loss_content_trg = self.mmd_criterion(trg_content_latent, fake_content_latent, 2) # z_var is 2 now

            # 1-3. Similarity Loss - src_content_latent and trg_content_latent
            sim_loss_content = self.content_similiarity_criterion(src_content_latent, trg_content_latent, torch.Tensor([1]).to(self.device)) #self.content_similiarity_criterion(src_content_latent, trg_content_latent).mean()

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            trg_style_latent = self.style_latent_encoder(encoder_out_trg) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            trg_style_latent = trg_style_latent.squeeze(2) # (batch_size, d_latent)

            # 2-2. WAE Process
            # 2-2-1. Draw fake style latent from predefined GMM distribution
            fake_style_latent = self.style_latent_gmm.sample((src_style_latent.shape[0],)) # (batch_size, d_latent)
            fake_style_latent = fake_style_latent.to(self.device)

            # 2-2-2. get mmd_loss between src_style_latent and fake_style_latent, trg_style_latent and fake_style_latent
            mmd_loss_style_src = self.mmd_criterion(src_style_latent, fake_style_latent, 0.5) # z_var is 0.5 now 
            mmd_loss_style_trg = self.mmd_criterion(trg_style_latent, fake_style_latent, 0.5) # Mixture dist에서 MMD가 제대로 작동하는지?

            # 2-3. Similarity Loss - src_style_latent and trg_style_latent
            sim_loss_style = self.style_similiarity_criterion(src_style_latent, trg_style_latent, torch.Tensor([-1]).to(self.device)) #self.style_similiarity_criterion(src_style_latent, trg_style_latent).mean()
            
            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_latent.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_latent.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 4. Define dist_loss
            dist_loss = mmd_loss_content_src + mmd_loss_content_trg + mmd_loss_style_src + mmd_loss_style_trg
            sim_loss = sim_loss_content + sim_loss_style # maximize sim_loss_content, minimize sim_loss_style
            
            dist_loss = dist_loss + sim_loss

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

        return encoder_out_total, dist_loss * self.loss_lambda

    def generate(self, encoder_out_src):

    #===================================#
    #================VAE================#
    #===================================#

        if self.variational_mode in [1,2]:
            src_mu = self.context_to_mu(encoder_out_src) # [token, batch, d_latent]
            resize_z = self.z_to_context(src_mu) # [token, batch, d_model]

            encoder_out_total = torch.add(encoder_out_src, resize_z)

    #===================================#
    #================WAE================#
    #===================================#

        if self.variational_mode in [3,4]:

            # Source sentence latent mapping
            src_latent = self.context_to_latent(encoder_out_src) # (token, batch, d_latent)
            src_latent = self.latent_to_context(src_latent)

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #==============CNN+VAE==============#
    #===================================#

        if self.variational_mode in [5,6]:

            src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
            src_mu = self.context_to_mu(src_latent) # [batch, d_latent]
            
            resize_z = self.latent_decoder(src_mu.unsqueeze(2)) # [seq_len, batch, d_model]

            encoder_out_total = torch.add(encoder_out_src, resize_z)
            
    #===================================#
    #==============CNN+WAE==============#
    #===================================#

        if self.variational_mode == [7,8]:

            src_latent = self.latent_encoder(encoder_out_src) # [batch, d_latent]
            src_latent = self.latent_decoder(src_latent.unsqueeze(2)) # [token, batch, d_model]

            encoder_out_total = torch.add(encoder_out_src, src_latent)

    #===================================#
    #======Gaussian Mixture + VAE=======#
    #===================================#

        """
        Need refactoring
        """

        if self.variational_mode == 9:

            # # 1. Style classifiction 
            # x_to_cls = encoder_out_src.mean(dim=0) # [batch, d_model]
            # distribution_cls = self.context_to_cls(x_to_cls) # [batch, num_cls]
            # distribution_cls_cp = distribution_cls.unsqueeze(0).repeat(encoder_out_src.size(0), 1, 1) # [token, batch, num_cls]

            # # concat
            # gm_inp = torch.cat((encoder_out_src, distribution_cls_cp), dim=2) # (tokn, batch, d_latent + num_cls)

            # src_mu = self.context_to_mu(gm_inp) # (token, batch, d_latent)
            # src_logvar = self.context_to_logvar(gm_inp) # (token, batch, d_latent)

            # mu = src_mu.view(gm_inp.size(1), -1) # [batch, seq_len * d_latent]
            # logvar = src_logvar.view(gm_inp.size(1), -1) # [batch, seq_len * d_latent]
            # dist_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

            # std = src_logvar.mul(0.5).exp_()
            # eps = Variable(std.data.new(std.size()).normal_())
            # z = eps.mul(std).add_(src_mu)

            # # 5. Decoding with 'z_to_context'
            # resize_z = self.z_to_context(z) # [batch, d_model]

            # encoder_out_total = torch.add(encoder_out_src, resize_z)

            raise Exception('Need refactoring...!! Try another variational mode')
            
    #===================================#
    #=========CNN+VAE(GMM&SIM)==========#
    #===================================#

        if self.variational_mode == 10:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)

            # 1-2. VAE Process
            # 1-2-1. Get mu and logvar from src_content_latent and trg_content_latent
            src_content_mu = self.content_latent_to_mu(src_content_latent) # (batch_size, d_latent)
            src_content_logvar = self.content_latent_to_logvar(src_content_latent) # (batch_size, d_latent)

            # 1-2-2. Reparameterization trick
            src_content_std = src_content_logvar.mul(0.5).exp_()
            src_content_eps = torch.randn_like(src_content_std).to(self.device)
            src_content_z = src_content_eps * src_content_std + src_content_mu # (batch_size, d_latent)

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)

            # 2-2. VAE Process
            # 2-2-1. Get mu and logvar from src_style_latent and trg_style_latent
            src_style_mu = self.style_latent_to_mu(src_style_latent) # (batch_size, d_latent)
            src_style_logvar = self.style_latent_to_logvar(src_style_latent) # (batch_size, d_latent)

            # 2-2-2. Reparameterization trick
            src_style_std = src_style_logvar.mul(0.5).exp_()
            src_style_eps = torch.randn_like(src_style_std).to(self.device)
            src_style_z = src_style_eps * src_style_std + src_style_mu # (batch_size, d_latent)

            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_z.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_z.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)


    #===================================#
    #=========CNN+WAE(GMM&SIM)==========#
    #===================================#
        
        if self.variational_mode == 11:
            # Source sentence latent mapping
            encoder_out_src = encoder_out_src.permute(1, 2, 0) # From: (seq_len, batch_size, d_model)

            # 1-1. Get content latent
            src_content_latent = self.content_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_content_latent = src_content_latent.squeeze(2) # (batch_size, d_latent)

            # 2-1. Get style latent
            src_style_latent = self.style_latent_encoder(encoder_out_src) # (batch_size, d_latent, 1)
            src_style_latent = src_style_latent.squeeze(2) # (batch_size, d_latent)
            
            # 3-1. Translate each src latent to d_model dimension
            src_content_latent = self.content_latent_decoder(src_content_latent.unsqueeze(2)) # (batch_size, d_model, 1)
            src_style_latent = self.style_latent_decoder(src_style_latent.unsqueeze(2)) # (batch_size, d_model, 1)

            # 3-2. add each src latent and repeat
            src_latent = src_content_latent + src_style_latent # (batch_size, d_model, 1)
            src_latent = src_latent.repeat(1, 1, encoder_out_src.size(2)) # (batch_size, d_model, seq_len)

            # 5. Get output
            encoder_out_total = torch.add(encoder_out_src, src_latent)
            encoder_out_total = encoder_out_total.permute(2, 0, 1) # (seq_len, batch_size, d_model)

        return encoder_out_total