def torch_manually():
    import torch
    device = torch.device('cpu')
    n, d_in, h, d_out = 64, 1000, 100, 10
    x = torch.randn(n, d_in, device=device)
    y = torch.randn(n, d_out, device=device)
    w1 = torch.randn(d_in, h, device=device)
    w2 = torch.randn(h, d_out, device=device)

    learning_rate = 1e-10
    for i in range(50):
        # forward
        h = x.mm(w1)
        h_r = h.clamp(min=0)
        y_pre = h_r.mm(w2)
        loss = (y-y_pre).pow(2).sum()
        # Backward (gradient)
        grad_pre = 1.0*(y_pre-y)
        grad_h_r = grad_pre.mm(w2.t())
        grad_h = grad_h_r.clone()
        grad_h[h < 0.01] = 0
        grad_w1 = x.t().mm(grad_h)
        grad_w2 = h_r.t().mm(grad_pre)
        w1 -= grad_w1*learning_rate
        w2 -= grad_w2*learning_rate


######################################################################################################################
def autograd_torch():
    import torch
    device = torch.device('cpu')
    n, d_in, h, d_out = 64, 1000, 100, 10
    x = torch.randn(n, d_in, device=device)
    y = torch.randn(n, d_out, device=device)
    # Each variable which needs gradient need to be mentioned
    w1 = torch.randn(d_in, h, device=device, requires_grad=True)
    w2 = torch.randn(h, d_out, device=device, requires_grad=True)

    learning_rate = 1e-6
    for i in range(500):
        # Forward
        y_pre = x.mm(w1).clamp(min=0).mm(w2)
        loss = (y-y_pre).pow(2).sum()
        # Backward
        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate*w1.grad
            w2 -= learning_rate*w2.grad
            w1.grad.zero_()
            w2.grad.zero_()
        print(loss)

#########################################################################
# Define new autograd function


def define_and_use_new_autograd_function():
    import torch
    # it is overriding, so it argument should be equal and
    # if you add one argument which is not in its parents (like x),
    # you should insert x=None

    class MyReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x=None, *args, **kwargs):
            ctx.save_for_backward(x)
            return x.clamp(min=0)

        @staticmethod
        def backward(ctx=None, grad_y=None, *grad_outputs):
            x, = ctx.saved_tensors
            grad_input = grad_y.clone()
            grad_input[x < 0] = 0
            return grad_input
    def my_relue(x):
        return MyReLU.apply(x)

    device = torch.device('cpu')
    n, d_in, h, d_out = 64, 1000, 100, 10
    x = torch.randn(n, d_in, device=device)
    y = torch.randn(n, d_out, device=device)
    # Each variable which needs gradient need to be mentioned
    w1 = torch.randn(d_in, h, device=device, requires_grad=True)
    w2 = torch.randn(h, d_out, device=device, requires_grad=True)

    learning_rate = 1e-6
    for i in range(500):
        y_pre = my_relue(x.mm(w1)).mm(w2)
        loss = (y-y_pre).pow(2).sum()
        # Backward
        loss.backward()

        with torch.no_grad():
            w1 -= learning_rate*w1.grad
            w2 -= learning_rate*w2.grad
            w1.grad.zero_()
            w2.grad.zero_()
        print(loss)
################################################################################


def torch_nn_without_opt():
    import torch
    device = torch.device('cpu')
    n, d_in, h, d_out = 64, 1000, 100, 10
    x = torch.randn(n, d_in, device=device)
    y = torch.randn(n, d_out, device=device)
    # it does not need the weights to be defined
    # Forward
    model = torch.nn.Sequential(torch.nn.Linear(d_in, h),
                                torch.nn.ReLU(),
                                torch.nn.Linear(h, d_out))
    learning_rate = 1e-2
    for t in range(500):
        y_pre = model(x)
        loss = torch.nn.functional.mse_loss(y_pre, y)
        loss.backward()

        with torch.no_grad():
            for par in model.parameters():
                par -= par.grad*learning_rate
        model.zero_grad()
        print(loss)
#######################################################################


def torch_nn_optim():
    import torch
    device = torch.device('cpu')
    n, d_in, h, d_out = 64, 1000, 100, 10
    x = torch.randn(n, d_in, device=device)
    y = torch.randn(n, d_out, device=device)
    # it does not need the weights to be defined
    model = torch.nn.Sequential(torch.nn.Linear(d_in, h),
                                torch.nn.ReLU(),
                                torch.nn.Linear(h, d_out))
    learnning_rate = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=learnning_rate)

    for t in range(500):
        y_pre = model(x)
        loss = torch.nn.functional.mse_loss(y_pre, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
#######################################################################


def define_new_nn_modules():
    import torch

    class TwoLayerNet(torch.nn.Module):
        def __init__(self, d_in, h, d_out):
            super().__init__()
            self.linear1=torch.nn.Linear(d_in, h)
            self.linear2=torch.nn.Linear(h,d_out)

        def forward(self,x):
            h_relue = self.linear1(x).clamp(min=0)
            y_pred = self.linear2(h_relue)
            return y_pred

    n, d_in, h, d_out = 64, 1000, 100,10
    x = torch.randn(n, d_in)
    y = torch.randn(n, d_out)

    model = TwoLayerNet(d_in, h, d_out)
    optimizer = torch.optim.SGD(params=model.parameters(), lr=1e-4)

    for t in range(500):
        y_pre = model(x)
        loss = torch.nn.functional.mse_loss(y_pre, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
########################################################################


def new_module_nn():
    import torch

    class parallelblock (torch.nn.Module):
        def __init__(self, d_in, d_out):
            super(parallelblock, self).__init__()
            self.linear1 = torch.nn.Linear(d_in,d_out)
            self.linear2 = torch.nn.Linear(d_in,d_out)

        def forward(self, x):
            h1 = self.linear1(x)
            h2 = self.linear2(x)
            return (h1*h2).clamp(min=0)
    n, d_in, h, d_out = 64, 1000, 100, 10
    x = torch.randn(n, d_in)
    y = torch.randn(n,d_out)

    model= torch.nn.Sequential(parallelblock(d_in,h),
                               parallelblock(h,h),
                               parallelblock(h, d_out))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for t in range(500):
        y_pred = model(x)
        loss = torch.nn.functional.mse_loss(y_pred,y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(loss)
#######################################################################

def data_loader():
    import torch

    class TwoLayerNet(torch.nn.Module):
        def __init__(self,d_in,h,d_out):
            super(TwoLayerNet, self).__init__()
            self.linear1=torch.nn.Linear(d_in,h)
            self.linear2=torch.nn.Linear(h,d_out)

        def forward(self,x):
            h_relue=self.linear1(x).clamp(min=0)
            y_pred=self.linear2(h_relue)
            return y_pred
    from torch.utils.data import TensorDataset, DataLoader
    n, d_in, h, d_out= 64, 1000, 100, 10
    x=torch.randn(n,d_in)
    y=torch.randn(n,d_out)

    loader= DataLoader(TensorDataset(x,y), batch_size=8)

    model =TwoLayerNet(d_in, h, d_out)
    optimizer=torch.optim.SGD(model.parameters(),lr=1e-3)

    for epoch in range(20):
        for x_batch, y_batch in loader:
            y_pre=model(x_batch)
            loss=torch.nn.functional.mse_loss(y_pre,y_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        print(loss)


# torch_manually()
# autograd_torch()
# define_and_use_new_autograd_function()
# torch_nn_without_opt()
# torch_nn_optim()
# define_new_nn_modules()
# new_module_nn()
data_loader()
