import torch 
from torch import nn 
import torch.nn.functional as F


class DownConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__() 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1)
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        
        out += self.shortcut(x)
        out = F.relu(out)

        pool = self.maxpool(out)
        return out, pool 

class UpConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=2, padding=0, stride=2) 
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, padding=1) 
        self.relu = nn.ReLU() 

        self.shortcut = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channel)
            )
    
    def forward(self, x, skip_connection_layer):
        up = self.upconv(x)
        up = torch.cat((skip_connection_layer, up), dim=1) 

        out = F.relu(self.conv1(up)) 
        out = F.relu(self.conv2(out))

        out += self.shortcut(up)
        out = F.relu(out) 

        return out


class UNet(nn.Module):
    def __init__(self, in_channel=3, num_classes=3):
        super().__init__()
        # encoder
        self.encoder1 = DownConv(in_channel, 64)
        self.encoder2 = DownConv(64, 128)
        self.encoder3 = DownConv(128, 256)
        self.encoder4 = DownConv(256, 512)
        
        # bottleneck 
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        
        # decoder 
        self.decoder1 = UpConv(1024, 512)
        self.decoder2 = UpConv(512, 256)
        self.decoder3 = UpConv(256, 128)
        self.decoder4 = UpConv(128, 64)

        # output 
        self.output = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, padding=0)

        
    def forward(self, x): 
        #ENCODER
        sc1, en1 = self.encoder1(x) #layer for skip connection, maxpool output
        sc2, en2 = self.encoder2(en1)
        sc3, en3 = self.encoder3(en2)
        sc4, en4 = self.encoder4(en3)
   
        #BOTTLENECK
        bottleneck = self.bottleneck(en4)
        
        #DECODER
        upconv1 = self.decoder1(bottleneck,sc4)
        upconv2 = self.decoder2(upconv1,sc3) 
        upconv3 = self.decoder3(upconv2, sc2)
        upconv4 = self.decoder4(upconv3, sc1)

        #OUTPUT 
        output = self.output(upconv4)
        return output


def get_model(): 
    return UNet() 


if __name__ == "__main__": 
    model = get_model()
    x = torch.rand(1, 3, 256, 256) 


    input_names = ["input"]
    output_names = ["output"]

    torch.onnx.export(
        model,
        x,
        "brain_unet.onnx",
        verbose=False,
        input_names=input_names,
        output_names=output_names,
        export_params=True,
    )
 