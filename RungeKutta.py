import numpy as np

class RK:

    def __init__(self,f,t_interv,x0,dt=10e-4,metodo='rk4',t_pasos=None,args=[]):
        self.f=f
        self.t0=t_interv[0]
        self.tf=t_interv[1]
        self.dt=dt
        self.x0=x0

        if t_pasos==None:
            t=np.arange(t_interv[0],t_interv[1]+dt,dt)
            N=len(t)
        else:
            t=t_pasos
            N=len(t)
        x = np.zeros((len(x0),N))
        x[:,0] = np.array(x0)

        if metodo=='rk4':
            x=self.rk4(t,x,args)
            self.t=t
            self.x=x
        elif metodo=='rk2':
            x=self.rk2(t,x,args)
            self.t=t
            self.x=x
        elif metodo=='rk1':
            x=self.rk1(t,x,args)
            self.t=t
            self.x=x

    def rk1(self,t,x,args):
        
        for i in range(len(t)-1):
            dt=t[i+1]-t[i]
            k_1 = self.f(t[i],x[:,i],*args)
            x[:,i+1] = x[:,i] + dt*k_1
    
        return x

    def rk2(self,t,x,args):
        
        for i in range(len(t)-1):
            dt=t[i+1]-t[i]
            k_1 = self.f(t[i],x[:,i],*args)
            k_2 = self.f(t[i],x[:,i]+dt*k_1,*args)
            x[:,i+1] = x[:,i] + (dt/2)*(k_1 + k_2)
    
        return x

    def rk4(self,t,x,args):
        
        for i in range(len(t)-1):
            dt=t[i+1]-t[i]
            k_1 = self.f(t[i],x[:,i],*args)
            k_2 = self.f(t[i],x[:,i]+0.5*dt*k_1,*args)
            k_3 = self.f(t[i],x[:,i]+0.5*dt*k_2,*args)
            k_4 = self.f(t[i],x[:,i]+dt*k_3,*args)
            x[:,i+1] = x[:,i] + (dt/6)*(k_1 + 2*k_2 + 2*k_3 + k_4)
    
        return x