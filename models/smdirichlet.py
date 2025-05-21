import torch
import torch.nn as nn
import torch.nn.functional as F

class SMDIRICHLET(nn.Module):
    
    @staticmethod
    def _param_initializer(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)

        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    def __init__(self, vocab_size, num_topics=10, hidden_size=256, hidden_layers=1, nonlinearity=nn.GELU):
        super().__init__()
        self.num_topics = num_topics
        self.vocab_size = vocab_size

        # First MLP layer compresses from vocab_size to hidden_size
        mlp_layers = [nn.Linear(vocab_size, hidden_size), nonlinearity()]
        # Remaining layers operate in dimension hidden_size
        for _ in range(hidden_layers - 1):
            mlp_layers.append(nn.Linear(hidden_size, hidden_size))
            mlp_layers.append(nonlinearity())

        self.mlp = nn.Sequential(*mlp_layers)
        self.mlp.apply(SMDIRICHLET._param_initializer)

        # Create linear projections for Gaussian params (rho1 & rho2)
        self.rho1 = nn.Linear(hidden_size, num_topics)
        self.rho1.apply(SMDIRICHLET._param_initializer)

        # Custom initialization for rho2
        self.rho2 = nn.Linear(hidden_size, num_topics)
        self.rho2.bias.data.zero_()
        self.rho2.weight.data.fill_(0.)

        # create linear projrction for alpha
        self.alpha = nn.Linear(hidden_size, num_topics)
        self.alpha.apply(SMDIRICHLET._param_initializer)

        self.dec_projection = nn.Linear(num_topics, vocab_size)
        self.log_softmax = nn.LogSoftmax(-1)

    def reparameterize(self, alpha, rho1, logrho2, eps):
        rho2 = torch.exp(logrho2)
        #eps = torch.randn_like(std)
        alpha_smoothed = alpha + eps * rho2 + rho1
       
        Z_sd =  F.softmax(alpha_smoothed, dim=1)

        return Z_sd, alpha_smoothed
    
    def kld(self, model_alpha, prior_alpha, epsilon): 

        model_alpha = torch.max(torch.tensor(0.0001), model_alpha).to(model_alpha.device)
        alpha = prior_alpha.expand_as(model_alpha)
        sum1 = torch.sum((model_alpha + epsilon - 1) * torch.digamma(model_alpha + epsilon), dim=1)

        sum2 = torch.sum((alpha + epsilon - 1) * torch.digamma(alpha + epsilon), dim=1)
        kl_loss = torch.mean(sum1 - sum2)

        return kl_loss 


    def forward(self, input_bows):
        # Run BOW through MLP
        pi = self.mlp(input_bows)

        # Use this to get mean, log_sig for Gaussian
        rho1 = self.rho1(pi)
        logrho2 = self.rho2(pi)
        alpha = self.alpha(pi)

        epsilons = torch.normal(0, 1, size=(
            input_bows.size()[0], self.num_topics)).to(input_bows.device)

        sample, alpha_smoothed  = self.reparameterize(alpha, rho1, logrho2, epsilons)

       
        # Softmax to get p(v_i | h_tm), AKA probabilities of words given hidden state
        logits = self.log_softmax(self.dec_projection(sample))

    
        kld = self.kld(alpha_smoothed , prior_alpha = torch.tensor(0.01), epsilon=torch.tensor(0.000000000001))
    
        rec_loss = -1 * torch.sum(logits * input_bows, 1)
        loss_Dir = torch.mean(rec_loss + kld)

        return sample, logits, torch.mean(kld), loss_Dir
