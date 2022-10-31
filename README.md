# Enhanced-Densnet-VIT
This is a Image classification by Enhanced Densnet-VIT
This is an image classification model based on VisionTransformer，
Firstly, Densnet is used to extract image features，
Another branch is transformed from vison transfomrer with STP, 
and another branch learns from the surface of the image through CBAM
Operation method：python tools/train.py models/myvit/myVIT.py
