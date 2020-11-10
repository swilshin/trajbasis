
from numpy import (sqrt,sum,tile,abs,logical_and,arctan2,sign,nan,ones_like,
  subtract,exp,array,pi,cos,vstack)

################################################################################
# BASIS FIT
################################################################################
class TrajectoryBasisSystem(object):
  '''
  A system for evaluating a weighting function for estimating angles with bases 
  at locations from a set of trajectories.
  '''
  def __init__(self,C):
    self.c0,self.c1 = C
    self.n = self.c1-self.c0
    self.u = sqrt(sum(self.n*self.n,1))
  
  def __call__(self,x,lda,thetas,wfunc,nw=1.0,xsub=None,eps0=1e-10):
    if xsub is None:
      if thetas.ndim==1:
        tau = tile(thetas,(x.shape[0],1))
      else:
        tau = thetas
      v = (
        tile(x,(self.c0.shape[0],1,1)).transpose(1,0,2)
        -
        tile(self.c0,(x.shape[0],1,1))
      )
      w = (
        tile(x,(self.c1.shape[0],1,1)).transpose(1,0,2)
        - 
        tile(self.c1,(x.shape[0],1,1))
      )
      p = sum([1,-1]*self.n[:,[1,0]]*v,-1)/self.u
      mu = sum(self.n*v,-1)/(self.u*self.u)
      d = nan*ones_like(mu)
      if sum(mu<=0)>0:
        d[mu<=0] = sqrt(sum(v[mu<=0]**2,-1))
      if sum(mu>=1)>0:
        d[mu>=1] = sqrt(sum(w[mu>=1]**2,-1))
      if sum(logical_and(mu<1,mu>0))>0:
        d[logical_and(mu<1,mu>0)] = sqrt(
          sum(v[logical_and(mu<1,mu>0)]*v[logical_and(mu<1,mu>0)],-1)
          -
          (mu*self.u)[logical_and(mu<1,mu>0)]**2
        )
      angs = subtract.outer(tau,arctan2(*self.n.T)).transpose(1,0,2)
      W0 = exp(-d*d/(2*lda*lda))
      W1 = wfunc(angs,p)
      WW = W1*W0
      W = (1.0/exp(exp(nw)))+sum(WW,-1).T
      return(W)
    else:
      W = None
      for i,j in zip(range(0,x.shape[0]+xsub,xsub)[:-1],range(0,x.shape[0]+xsub,xsub)[1:]):
        xs = x[i:j]
        if thetas.ndim==1:
          tau = tile(thetas,(xs.shape[0],1))
        else:
          tau = thetas[i:j]
        v = (
          tile(xs,(self.c0.shape[0],1,1)).transpose(1,0,2)
          -
          tile(self.c0,(xs.shape[0],1,1))
        )
        w = (
          tile(xs,(self.c1.shape[0],1,1)).transpose(1,0,2)
          - 
          tile(self.c1,(xs.shape[0],1,1))
        )
        p = sum([1,-1]*self.n[:,[1,0]]*v,-1)/self.u
        mu = sum(self.n*v,-1)/(self.u*self.u)
        d = nan*ones_like(mu)
        if sum(mu<=0)>0:
          d[mu<=0] = sqrt(sum(v[mu<=0]**2,-1))
        if sum(mu>=1)>0:
          d[mu>=1] = sqrt(sum(w[mu>=1]**2,-1))
        if sum(logical_and(mu<1,mu>0))>0:
          kai = (
            sum(v[logical_and(mu<1,mu>0)]*v[logical_and(mu<1,mu>0)],-1)
            -
            (mu*self.u)[logical_and(mu<1,mu>0)]**2
          )
          kai[abs(kai)<eps0] = 0.0
          d[logical_and(mu<1,mu>0)] = sqrt(kai)
        angs = subtract.outer(tau,arctan2(*self.n.T)).transpose(1,0,2)
        W0 = exp(-d*d/(2*lda*lda))
        W1 = wfunc(angs,p)
        WW = W1*W0
        Ws = (1.0/exp(exp(nw)))+sum(WW,-1).T
        if W is None:
          W = Ws
        else:
          W = vstack([W,Ws])
      return(W)

# Basis functions
def adjustedBasis(t,p,s0=0.02,gamma=0.1):
  pp = (2.*array(p>=0,dtype=float)-1.)*(1.+exp(-(p*p)/(2*s0*s0)))/2.0
  alpha = abs(pp)
  q = sign(pp)
  b = ((2*alpha+q)/(2*alpha))*pi
  c = 2*pi*(1-alpha)
  tau0 = (t - q*((pi/2.) - (pi/(2.*alpha))))%(2*pi)
  taup = alpha*tau0+c*(tau0>b)
  return(1+gamma+cos(2*taup)+gamma*cos(6*taup))

def adjustedVMBasis(t,p,s0=0.02,gamma=0.0):
  pp = (2.*array(p>=0,dtype=float)-1.)*(1.+exp(-(p*p)/(2*s0*s0)))/2.0
  alpha = abs(pp)
  q = sign(pp)
  b = ((2*alpha+q)/(2*alpha))*pi
  c = 2*pi*(1-alpha)
  tau0 = (t - q*((pi/2.) - (pi/(2.*alpha))))%(2*pi)
  taup = alpha*tau0+c*(tau0>b)
  return(exp((1+gamma)*cos(2*taup)))
