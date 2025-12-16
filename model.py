import torch
from torchmil.nn import masked_softmax
from torchvision.models import resnet18, ResNet18_Weights
from timm.models import create_model


class AttentionMILModel(torch.nn.Module):
    def __init__(self, output_dim, att_dim, emb_dim=768):
        super().__init__()

        self.fc1 = torch.nn.Linear(emb_dim, att_dim)
        self.fc2 = torch.nn.Linear(att_dim, 1)

        self.classifier = torch.nn.Linear(emb_dim, output_dim)

        self.dropout = torch.nn.Dropout(p=0.5)

    def forward(self, X, mask, bag_size, feature_extractor, return_att=False):
        batch_size = int(X.shape[0] / bag_size)

        # Process only instances that are not masked (i.e., valid instances, not padding)
        with torch.no_grad():
            X = feature_extractor(X[mask != 0]).detach()  # (batch_size * bag_size, emb_dim)

        # Put back the processed instances to their original positions, so that the shape is preserved (as if all instances, including padding, were processed)
        fe_output = torch.zeros((batch_size * bag_size, X.shape[1]), device=X.device)
        fe_output[mask != 0] = X
        X = fe_output

        # Reshaping to separate bags from batches
        X = X.reshape((batch_size, bag_size, -1))  # (batch_size, bag_size, emb_dim)
        mask = mask.reshape((batch_size, bag_size))  # (batch_size, bag_size)

        H = torch.tanh(self.fc1(X))  # (batch_size, bag_size, att_dim)
        att = torch.sigmoid(self.fc2(H))  # (batch_size, bag_size, 1)

        att_s = masked_softmax(att, mask)  # (batch_size, bag_size, 1)
        # att_s = torch.nn.functional.softmax(att, dim=1)
        X = torch.bmm(att_s.transpose(1, 2), X).squeeze(1)  # (batch_size, emb_dim)
        X = self.dropout(X)
        y = self.classifier(X).squeeze(1)  # (batch_size,)
        if return_att:
            return y, att_s
        else:
            return y


class FeatureExtractor(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.feature_extractor = create_model(
                'convnext_small.fb_in22k_ft_in1k_384',
                num_classes=1,
                in_chans=3,
                pretrained=False,
                checkpoint_path="rsna-breast-cancer-detection-best-ckpts/best_convnext_fold_3.pth.tar",
                global_pool='max',
            )
        
        emb_dim = self.feature_extractor.head.fc.in_features
        self.feature_extractor.head.fc = torch.nn.Identity()

    def forward(self, X):
        return self.feature_extractor(X)