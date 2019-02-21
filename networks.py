import torch
import torch.nn as nn 

def conv(ic, oc, k, s, p):
    return nn.Sequential(
        nn.Conv2d(ic, oc, k, s, p),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(oc),
    )

class FeatureExtractor(nn.Module):
    def __init__(self, filters):
        super(FeatureExtractor, self).__init__()
        
        layers = []
        
        for in_channels, out_channels, kernel_size, stride, padding in filters:
            layers.append(conv(in_channels, out_channels, kernel_size, stride, padding))
            self.out_channels = out_channels            
        
        self.model = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.model(x)
    
    
class ModelG(nn.Module):
    def __init__(self, channels):
        super(ModelG, self).__init__()
        
        channels[0] = (channels[0] + 2) * 2 + 11

        layers = []
        for in_plane, out_plane in zip(channels[:-1], channels[1:]):
            layers.append(nn.Linear(in_plane, out_plane))
            layers.append(nn.ReLU(inplace=True))
                     
        self.out_channels = channels[-1]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)
    

class ModelF(nn.Module):
    def __init__(self, channels):
        super(ModelF, self).__init__()

        layers = []
        for in_plane, out_plane in zip(channels[:-1], channels[1:]):
            layers.append(nn.Linear(in_plane, out_plane))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Dropout(p=0.5))
        layers.append(nn.Linear(channels[-1], 10))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
        
        
class RN(nn.Module):
    def __init__(self, feature_extractor, g, f):
        super(RN, self).__init__()
        self.feature_extractor = feature_extractor
        self.g = g
        self.f = f
        
    def forward(self, image, question):
        x = self.feature_extractor(image)
        batch_size = x.size(0)
        k = x.size(1)
        d = x.size(2)
        
        # tag arbitrary coordinate
        coordinate = torch.arange(0, 1 + 0.00001, 1 / (d-1)).cuda()
        coordinate_x = coordinate.expand(batch_size, 1, d, d)
        coordinate_y = coordinate.view(d, 1).expand(batch_size, 1, d, d)
        x = torch.cat([x, coordinate_x, coordinate_y], 1)
        k += 2
        
        x = x.view(batch_size, k, d ** 2).permute(0, 2, 1)
        
        # concatnate o_i, o_j and q
        x_left = x.unsqueeze(1).repeat(1, d ** 2, 1, 1).view(batch_size, d ** 4, k)
        x_right = x.unsqueeze(2).repeat(1, 1, d ** 2, 1).view(batch_size, d ** 4, k)
        x = torch.cat([x_left, x_right], 2)        
        
        question = question.unsqueeze(1).repeat(1, d ** 4, 1)
        
        x = torch.cat([x, question], 2)
        x = x.view(batch_size * (d ** 4), k * 2 + 11)
        
        # g(o_i, o_j, q)
        x = self.g(x)
        x = x.view(batch_size, d ** 4, x.size(1))
        # Σg(o_i, o_j, q)
        x = torch.sum(x, dim=1)
        # f(Σg(o_i, o_j, q))
        x = self.f(x)
        
        return x
    
    
def make_model(model_dict):
    if model_dict['conv'] == 'light':
        feature_extractor = FeatureExtractor([
            (3, 24, 3, 2, 1),
            (24, 24, 3, 2, 1),
            (24, 24, 3, 2, 1),
            (24, 24, 3, 2, 1),
        ])
    elif model_dict['conv'] == 'heavy':
        feature_extractor = FeatureExtractor([
            (3, 32, 3, 2, 1),
            (32, 64, 3, 2, 1),
            (64, 128, 3, 2, 1),
            (128, 256, 3, 2, 1),
        ])
    elif model_dict['conv'] == 'patch':
        feature_extractor = FeatureExtractor([
            (3, 24, 3, 2, 1),
            (24, 24, 3, 2, 1),
            (24, 24, 3, 2, 0),
            (24, 24, 3, 1, 1),
            (24, 24, 3, 1, 0),
        ])
    
    prev_out_channels = feature_extractor.out_channels
    if model_dict['g'] == 'light':
        g = ModelG([prev_out_channels, 256, 256, 256, 256])
    elif model_dict['g'] == 'heavy':
        g = ModelG([prev_out_channels, 2000, 2000, 2000, 2000])
        
    prev_out_channels = g.out_channels
    if model_dict['f'] == 'light':
        f = ModelF([prev_out_channels, 256, 256])
    elif model_dict['f'] == 'heavy':
        f = ModelF([prev_out_channels, 2000, 1000, 500, 100])
        
    return RN(feature_extractor, g, f)