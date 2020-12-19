import torch.nn as nn


# define model
class TexasClassifier(nn.Module):
    def __init__(self,num_classes=100):
        super(TexasClassifier, self).__init__()

        self.features = nn.Sequential(
            nn.Linear(6169,1024),
            nn.Tanh(),
            nn.Linear(1024,512),
            nn.Tanh(),
            nn.Linear(512,256),
            nn.Tanh(),
            nn.Linear(256,128),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(128,num_classes)
        
    def forward(self,x):
        hidden_out = self.features(x)
        return self.classifier(hidden_out)

def texas(**kwargs):
    
    model = TexasClassifier(**kwargs)
    return model