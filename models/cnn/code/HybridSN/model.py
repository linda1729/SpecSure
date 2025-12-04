import torch
import torch.nn as nn

class HybridSN(nn.Module):
    def __init__(self, window_size, K, output_units):
        super().__init__()
        self.conv3d_1 = nn.Conv3d(1,8,(7,3,3))
        self.conv3d_2 = nn.Conv3d(8,16,(5,3,3))
        self.conv3d_3 = nn.Conv3d(16,32,(3,3,3))
        self.relu = nn.ReLU(inplace=True)
        self.conv2d = None
        self.flatten = nn.Flatten()
        self.fc1 = None
        self.drop1 = nn.Dropout(0.4)
        self.fc2 = nn.Linear(256,128)
        self.drop2 = nn.Dropout(0.4)
        self.fc_out = nn.Linear(128, output_units)
        self._built = False
    def _build_layers(self, x):
        b,c,d,h,w = x.shape
        in_ch = c*d
        self.conv2d = nn.Conv2d(in_ch,64,3)
        out_h = h-2
        out_w = w-2
        self.fc1 = nn.Linear(64*out_h*out_w,256)
        self._built = True
    def forward(self,x):
        x = self.relu(self.conv3d_1(x))
        x = self.relu(self.conv3d_2(x))
        x = self.relu(self.conv3d_3(x))
        if not self._built:
            self._build_layers(x)
            self.conv2d.to(x.device)
            self.fc1.to(x.device)
        b,c,d,h,w = x.shape
        x = x.view(b,c*d,h,w)
        x = self.relu(self.conv2d(x))
        x = self.flatten(x)
        x = self.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.relu(self.fc2(x))
        x = self.drop2(x)
        x = self.fc_out(x)
        return x
