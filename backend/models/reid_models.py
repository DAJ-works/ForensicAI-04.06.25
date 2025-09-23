import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union, Any

class ConvLayer(nn.Module):
    """Basic convolutional layer."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        IN: bool = False  # Instance Normalization
    ):
        super(ConvLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=False, groups=groups
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.IN = None
        if IN:
            self.IN = nn.InstanceNorm2d(out_channels, affine=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.IN is not None:
            x = self.IN(x)
        x = self.relu(x)
        return x

class OSBlock(nn.Module):
    """Omni-scale residual block."""
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        bottleneck_reduction: int = 4,
        IN: bool = False,
        stride: int = 1,
        downsample: Optional[nn.Module] = None
    ):
        super(OSBlock, self).__init__()
        mid_channels = out_channels // bottleneck_reduction
        
        self.conv1 = ConvLayer(in_channels, mid_channels, 1, IN=IN)
        
        # Multi-scale feature learning via parallel conv filters of different scales
        self.conv2a = ConvLayer(mid_channels, mid_channels//4, 3, padding=1, IN=IN)
        self.conv2b = ConvLayer(mid_channels//4, mid_channels//4, 3, padding=1, IN=IN)
        self.conv2c = ConvLayer(mid_channels//4, mid_channels//4, 3, padding=1, IN=IN)
        self.conv2d = ConvLayer(mid_channels//4, mid_channels//4, 3, padding=1, IN=IN)
        
        self.conv3 = ConvLayer(mid_channels, out_channels, 1, IN=IN)
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels or stride > 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        # First 1x1 conv
        x1 = self.conv1(x)
        
        # Multi-scale branches
        x2a = self.conv2a(x1)
        x2b = self.conv2b(x2a)
        x2c = self.conv2c(x2b)
        x2d = self.conv2d(x2c)
        
        # Concatenate multi-scale features
        x2 = torch.cat([x2a, x2b, x2c, x2d], dim=1)
        
        # Last 1x1 conv
        x3 = self.conv3(x2)
        
        # Skip connection
        out = x3 + self.shortcut(residual)
        
        # Final activation
        out = F.relu(out, inplace=True)
        
        return out

class OSNet(nn.Module):
    """
    Omni-Scale Network for person re-identification.
    Reference: 
    - Zhou et al. "Omni-Scale Feature Learning for Person Re-Identification" ICCV 2019
    """
    
    def __init__(
        self,
        num_classes: int = 1000,
        blocks: List[int] = [2, 2, 2],
        channels: List[int] = [64, 256, 384, 512],
        feature_dim: int = 512,
        dropout_p: float = 0.0,
        use_attention: bool = True
    ):
        super(OSNet, self).__init__()
        self.feature_dim = feature_dim
        
        # Stem layers
        self.conv1 = ConvLayer(3, channels[0], 7, stride=2, padding=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # OSNet blocks
        self.layers = nn.ModuleList()
        
        # Conv stage 1
        in_channels = channels[0]
        layer1 = []
        for i in range(blocks[0]):
            layer1.append(OSBlock(in_channels, channels[1]))
            in_channels = channels[1]
        self.layers.append(nn.Sequential(*layer1))
        
        # Transition layer 1
        self.transition1 = ConvLayer(channels[1], channels[1], 1)
        
        # Conv stage 2
        layer2 = []
        for i in range(blocks[1]):
            layer2.append(OSBlock(channels[1], channels[2]))
            in_channels = channels[2]
        self.layers.append(nn.Sequential(*layer2))
        
        # Transition layer 2
        self.transition2 = ConvLayer(channels[2], channels[2], 1)
        
        # Conv stage 3
        layer3 = []
        for i in range(blocks[2]):
            layer3.append(OSBlock(channels[2], channels[3]))
            in_channels = channels[3]
        self.layers.append(nn.Sequential(*layer3))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature layer and classifier
        self.feature_layer = nn.Linear(channels[3], feature_dim)
        self.classifier = nn.Linear(feature_dim, num_classes)
        
        # Attention module (optional)
        self.use_attention = use_attention
        if use_attention:
            self.attention = nn.Sequential(
                nn.Conv2d(channels[3], channels[3] // 16, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels[3] // 16, channels[3], 1),
                nn.Sigmoid()
            )
        
        # Dropout
        self.dropout = nn.Dropout(p=dropout_p)
        
        # Initialize weights
        self._init_params()
    
    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def featuremaps(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        
        for layer in self.layers:
            x = layer(x)
        
        return x
    
    def forward(self, x, return_featuremaps=False):
        # Feature maps
        x = self.featuremaps(x)
        
        if return_featuremaps:
            return x
        
        # Apply attention if enabled
        if self.use_attention:
            attention = self.attention(x)
            x = x * attention
        
        # Global pooling
        v = self.global_pool(x)
        v = v.view(v.size(0), -1)
        
        # Feature embedding
        v = self.feature_layer(v)
        
        # Classification output
        if not self.training:
            return v
        
        y = self.dropout(v)
        y = self.classifier(y)
        
        return y, v


class MGNModel(nn.Module):
    """
    Multiple Granularity Network (MGN) for person re-identification.
    Reference:
    - Wang et al. "Learning Discriminative Features with Multiple Granularities for Person Re-Identification" MM 2018
    """
    
    def __init__(
        self,
        num_classes: int = 751,
        feature_dim: int = 2048,
        pretrained: bool = True
    ):
        super(MGNModel, self).__init__()
        self.feature_dim = feature_dim
        
        # Load ResNet50 backbone
        resnet50 = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
        
        # ResNet50 backbone (first part)
        self.backbone = nn.Sequential(
            resnet50.conv1,
            resnet50.bn1,
            resnet50.relu,
            resnet50.maxpool,
            resnet50.layer1,
            resnet50.layer2,
            resnet50.layer3[0]
        )
        
        # Part-1: Global Branch
        self.global_branch = nn.Sequential(
            resnet50.layer3[1:],
            resnet50.layer4
        )
        
        # Part-2: 2-parts Branch
        self.part2_branch = nn.Sequential(
            copy_layer(resnet50.layer3[1:]),
            copy_layer(resnet50.layer4)
        )
        
        # Part-3: 3-parts Branch
        self.part3_branch = nn.Sequential(
            copy_layer(resnet50.layer3[1:]),
            copy_layer(resnet50.layer4)
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.global_pool_p2 = nn.AdaptiveAvgPool2d((2, 1))
        self.global_pool_p3 = nn.AdaptiveAvgPool2d((3, 1))
        
        # Feature projection layers
        self.global_reduction = nn.Sequential(
            nn.Conv2d(2048, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.part2_reduction = nn.Sequential(
            nn.Conv2d(2048, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        self.part3_reduction = nn.Sequential(
            nn.Conv2d(2048, feature_dim, 1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(inplace=True)
        )
        
        # Classifiers
        self.global_classifier = nn.Linear(feature_dim, num_classes)
        self.part2_classifier = nn.ModuleList([
            nn.Linear(feature_dim, num_classes),
            nn.Linear(feature_dim, num_classes)
        ])
        self.part3_classifier = nn.ModuleList([
            nn.Linear(feature_dim, num_classes),
            nn.Linear(feature_dim, num_classes),
            nn.Linear(feature_dim, num_classes)
        ])
        
        # Initialize classifiers
        for classifier in [self.global_classifier] + list(self.part2_classifier) + list(self.part3_classifier):
            nn.init.normal_(classifier.weight, std=0.001)
            nn.init.constant_(classifier.bias, 0)
    
    def forward(self, x):
        # Common feature extraction
        x = self.backbone(x)
        
        # Global branch
        g_feat = self.global_branch(x)
        g_pool = self.global_pool(g_feat)
        g_feat_reduced = self.global_reduction(g_pool).squeeze(3).squeeze(2)
        
        # Part-2 branch
        p2_feat = self.part2_branch(x)
        p2_pool = self.global_pool_p2(p2_feat)
        p2_feat_reduced = self.part2_reduction(p2_pool)
        p2_features = [p2_feat_reduced[:, :, i, :].squeeze(2) for i in range(2)]
        
        # Part-3 branch
        p3_feat = self.part3_branch(x)
        p3_pool = self.global_pool_p3(p3_feat)
        p3_feat_reduced = self.part3_reduction(p3_pool)
        p3_features = [p3_feat_reduced[:, :, i, :].squeeze(2) for i in range(3)]
        
        # If in evaluation mode, return concatenated features
        if not self.training:
            features = torch.cat([
                g_feat_reduced,
                *p2_features,
                *p3_features
            ], dim=1)
            return features
        
        # If in training mode, return classification outputs and features
        g_cls = self.global_classifier(g_feat_reduced)
        p2_cls = [classifier(feature) for classifier, feature in zip(self.part2_classifier, p2_features)]
        p3_cls = [classifier(feature) for classifier, feature in zip(self.part3_classifier, p3_features)]
        
        return {
            'global': {
                'feature': g_feat_reduced,
                'logits': g_cls
            },
            'part2': {
                'features': p2_features,
                'logits': p2_cls
            },
            'part3': {
                'features': p3_features,
                'logits': p3_cls
            }
        }

def copy_layer(layer):
    """Helper function to copy a layer with its parameters"""
    if isinstance(layer, nn.Sequential):
        return nn.Sequential(*[copy_layer(l) for l in layer])
    else:
        copied_layer = type(layer)(*(param.clone() for param in layer.parameters))
        for param, copied_param in zip(layer.parameters(), copied_layer.parameters()):
            copied_param.detach_()
            copied_param.requires_grad_(True)
            copied_param.copy_(param.data)
        return copied_layer


class AttributePredictor(nn.Module):
    """
    Attribute prediction module for person re-identification.
    Predicts clothing attributes, gender, accessories, etc.
    """
    
    def __init__(
        self,
        feature_dim: int = 2048,
        attribute_dims: Dict[str, int] = {
            'gender': 2,
            'age': 4,
            'upper_color': 10,
            'lower_color': 10,
            'upper_type': 8,
            'lower_type': 6,
            'hair': 3,
            'accessories': 6
        }
    ):
        super(AttributePredictor, self).__init__()
        self.attribute_dims = attribute_dims
        
        # Create separate classifiers for each attribute
        self.classifiers = nn.ModuleDict()
        for attr_name, attr_dim in attribute_dims.items():
            self.classifiers[attr_name] = nn.Sequential(
                nn.Linear(feature_dim, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(inplace=True),
                nn.Dropout(0.5),
                nn.Linear(512, attr_dim)
            )
    
    def forward(self, features):
        """
        Forward pass through attribute predictor.
        
        Parameters:
        -----------
        features : torch.Tensor
            Feature vectors from base network
            
        Returns:
        --------
        Dict[str, torch.Tensor]
            Dictionary of attribute predictions
        """
        predictions = {}
        for attr_name, classifier in self.classifiers.items():
            predictions[attr_name] = classifier(features)
        
        return predictions