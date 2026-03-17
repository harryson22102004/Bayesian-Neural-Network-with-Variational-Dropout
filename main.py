import torch, torch.nn as nn, torch.nn.functional as F
 
class BayesLinear(nn.Module):
    def __init__(self, in_f, out_f):
        super().__init__()
        self.w_mu=nn.Parameter(torch.zeros(out_f,in_f)); nn.init.kaiming_uniform_(self.w_mu)
        self.w_ls=nn.Parameter(torch.full((out_f,in_f),-5.))
        self.b_mu=nn.Parameter(torch.zeros(out_f))
    def forward(self, x, sample=True):
        w = self.w_mu+torch.exp(self.w_ls)*torch.randn_like(self.w_mu) if sample else self.w_mu
        return F.linear(x,w,self.b_mu)
    def kl(self):
        s=torch.exp(self.w_ls)
        return -0.5*torch.sum(1+2*self.w_ls-self.w_mu**2-s**2)
 
class BNN(nn.Module):
    def __init__(self, in_d=2, h=64, out_d=1):
        super().__init__(); self.l1=BayesLinear(in_d,h); self.l2=BayesLinear(h,out_d)
    def forward(self,x,sample=True): return self.l2(F.relu(self.l1(x,sample)),sample)
    def elbo(self,x,y,n=5,N=1000):
        ll=torch.stack([-F.mse_loss(self(x).squeeze(),y) for _ in range(n)]).mean()
        return -(ll-(self.l1.kl()+self.l2.kl())/N)
 
model=BNN(2,32,1)
x=torch.randn(200,2); y=x[:,0]*2+x[:,1]+torch.randn(200)*0.1
opt=torch.optim.Adam(model.parameters(),1e-3)
for ep in range(300):
    loss=model.elbo(x,y); opt.zero_grad(); loss.backward(); opt.step()
preds=torch.stack([model(x[:5]) for _ in range(100)])
print(f"ELBO loss: {loss.item():.3f} | Predictive std: {preds.std(0).mean().item():.4f}")
