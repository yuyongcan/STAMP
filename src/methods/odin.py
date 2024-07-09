import torch.nn as nn
import torch
loss_fn = nn.CrossEntropyLoss()
class ODIN(nn.Module):
    def __init__(self, model, temperature, magnitude):
        super(ODIN, self).__init__()
        self.model = self.configure_model(model)
        self.temperature = temperature
        self.magnitude = magnitude
    @staticmethod
    def configure_model(model):
        model.train()
        for p in model.parameters():
            p.requires_grad = False
        return model
    @torch.enable_grad()
    def forward(self, x):
        x.requires_grad = True
        output_1 = self.model(x)
        maxIndexTemp = torch.argmax(output_1, dim=1)
        output_1 = output_1 / self.temperature
        loss = loss_fn(output_1, maxIndexTemp)
        loss.backward()
        with torch.no_grad():
            x_adv = x + self.magnitude * torch.sign(x.grad)
        output = self.model(x_adv)
        return output_1, output.max(1)[0]
