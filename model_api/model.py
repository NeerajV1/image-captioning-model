import torch
import torch.nn as nn
from torchvision import transforms
from transformers import CLIPVisionModel
device="cuda" if torch.cuda.is_available() else "cpu"
checkpoint=torch.load(r"D:\Neeraj\2026\Projects\image-caption-generator\models\clip_captioner_v2.pth",map_location=device)
word2idx=checkpoint["word2idx"]
idx2word=checkpoint["idx2word"]
class CLIPEncoder(nn.Module):
    def __init__(self, model_name="openai/clip-vit-base-patch32", freeze=True):
        super(CLIPEncoder, self).__init__()
        self.clip_vision = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze CLIP weights to retain original vision knowledge
        if freeze:
            for param in self.clip_vision.parameters():
                param.requires_grad = False

    def forward(self, images):
        """
        Input: [batch_size, 3, 224, 224]
        Output: [batch_size, 50, 768] (50 patches, 768 hidden feature dimension)
        """
        outputs = self.clip_vision(pixel_values=images)
        return outputs.last_hidden_state


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden).unsqueeze(1)
        att = self.full_att(self.relu(att1 + att2)).squeeze(2)
        alpha = self.softmax(att)
        context = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return context, alpha


class DecoderWithAttention(nn.Module):
    def __init__(self, attention_dim, embed_dim, decoder_dim, vocab_size, encoder_dim=768):
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.attention = BahdanauAttention(encoder_dim, decoder_dim, attention_dim)
        self.init_h = nn.Linear(encoder_dim, decoder_dim)
        self.init_c = nn.Linear(encoder_dim, decoder_dim)
        self.lstm_cell = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim)
        self.fc = nn.Linear(decoder_dim, vocab_size)

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, captions):
        batch_size = encoder_out.size(0)
        seq_len = captions.size(1)

        embeddings = self.embedding(captions)
        h, c = self.init_hidden_state(encoder_out)

        predictions = torch.zeros(batch_size, seq_len, self.vocab_size, device=encoder_out.device)

        for t in range(seq_len):
            context, _ = self.attention(encoder_out, h)
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.lstm_cell(lstm_input, (h, c))
            preds = self.fc(h)
            predictions[:, t, :] = preds

        return predictions


class CLIPImageCaptioner(nn.Module):
    def __init__(self, attention_dim=512, embed_dim=256, decoder_dim=512, vocab_size=len(word2idx)):
        super(CLIPImageCaptioner, self).__init__()
        self.encoder = CLIPEncoder(freeze=True)
        self.decoder = DecoderWithAttention(
            attention_dim=attention_dim,
            embed_dim=embed_dim,
            decoder_dim=decoder_dim,
            vocab_size=vocab_size,
            encoder_dim=768  # Matches CLIP ViT hidden dimension
        )

    def forward(self, images, captions):
        encoder_out = self.encoder(images)
        outputs = self.decoder(encoder_out, captions)
        return outputs
clip_model=CLIPImageCaptioner(
    attention_dim=512,
    embed_dim=256,
    decoder_dim=512,
    vocab_size=len(word2idx)
)
if hasattr(clip_model,"module"):
    clip_model.module.load_state_dict("model_state_dict")
else:
    clip_model.load_state_dict("model_state_dict")
clip_model.to(device)
