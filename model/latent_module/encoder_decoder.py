import torch
import torch.nn as nn

class full_cnn_latent_encoder(nn.Module):
    def __init__(self, d_model, d_latent, token_length):

        super(full_cnn_latent_encoder, self).__init__()
        if token_length == 100:
            self.encoder = nn.Sequential(
                        nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3), # (batch_size, d_model, 33)
                        nn.GELU(),
                        nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=3, stride=3), # (batch_size, d_model, 11)
                        nn.GELU(),
                        nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3), # (batch_size, d_model, 3)
                        nn.GELU(),
                        nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 1)
                        nn.GELU()
                    )
        if token_length == 200:
            self.encoder = nn.Sequential(
                        nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3), # (batch_size, d_model, 66)
                        nn.GELU(),
                        nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=3, stride=3), # (batch_size, d_model, 22)
                        nn.GELU(),
                        nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3), # (batch_size, d_model, 7)
                        nn.GELU(),
                        nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 2)
                        nn.GELU(),
                        nn.Conv1d(in_channels=d_latent, out_channels=d_latent, kernel_size=2, stride=3), # (batch_size, d_model, 1)
                        nn.GELU()
                    )
        if token_length == 300:
            self.encoder = nn.Sequential(
                        nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3), # (batch_size, d_model, 100)
                        nn.GELU(),
                        nn.Conv1d(in_channels=d_model, out_channels=512, kernel_size=3, stride=3), # (batch_size, d_model, 33)
                        nn.GELU(),
                        nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=3), # (batch_size, d_model, 11)
                        nn.GELU(),
                        nn.Conv1d(in_channels=256, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 3)
                        nn.GELU(),
                        nn.Conv1d(in_channels=d_latent, out_channels=d_latent, kernel_size=3, stride=3), # (batch_size, d_model, 1)
                        nn.GELU()
                    )
        else:
            raise Exception('Sorry, Now only 100, 200, 300 length is available')

    def forward(self, src):
        src = src.permute(1, 2, 0) # (batch, d_model, seq_len)
        latent = self.encoder(src) # (batch, d_latent, 1)
        latent = latent.squeeze(2)
        
        return latent

class full_cnn_latent_decoder(nn.Module):
    def __init__(self, d_model, d_latent, token_length):

        super(full_cnn_latent_decoder, self).__init__()
        if token_length == 100:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=256, kernel_size=10, stride=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=512, out_channels=d_model, kernel_size=3, stride=3, output_padding=1),
                nn.GELU(),
            )
        if token_length == 200:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=256, kernel_size=10, stride=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=512, out_channels=d_model, kernel_size=3, stride=3, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=2, stride=2),
                nn.GELU()
            )
        if token_length == 300:
            self.decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=256, kernel_size=10, stride=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=256, out_channels=512, kernel_size=5, stride=3, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=512, out_channels=d_model, kernel_size=3, stride=3, output_padding=1),
                nn.GELU(),
                nn.ConvTranspose1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=3),
                nn.GELU()
            )
        else:
            raise Exception('Sorry, Now only 100, 200, 300 length is available')

    def forward(self, src):
        latent = self.decoder(src) # [batch, d_model, seq_len]
        latent = latent.permute(2, 0, 1) # [seq_len, batch, d_model]

        return latent

class cnn_latent_encoder(nn.Module):
    def __init__(self, d_model, d_latent):

        super(cnn_latent_encoder, self).__init__()

        self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=d_model, out_channels=d_latent, kernel_size=3, stride=1, bias=True),
                nn.ReLU(inplace=False)
            )

    def forward(self, src):
        src = src.permute(1, 2, 0) # [batch, d_model, seq_len]
        latent = self.encoder(src) # [batch, d_latent, seq_len - 2]
        latent, _ = torch.max(latent, dim=2)

        return latent

class cnn_latent_decoder(nn.Module):
    def __init__(self, d_model, d_latent):

        super(cnn_latent_decoder, self).__init__()

        self.decoder = nn.Sequential(
                nn.ConvTranspose1d(in_channels=d_latent, out_channels=d_model, kernel_size=1, stride=1),
                nn.ReLU(inplace=False)
            )

    def forward(self, src):
        latent = self.decoder(src) # [batch, d_model, 1]
        latent = latent.permute(2, 0, 1) # [1, batch, d_model]

        return latent