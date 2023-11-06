class Model(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()

        # 선학습 모델에서는 특징 추출 부분만 사용하고 
        # (N,3,380,280)->(N,1792,12,12)
        self.features = backbone
        
        # (N,1792,12,12)->(N,1792,1,1)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(output_size=(1, 1))

        self.classifier = torch.nn.Sequential(
            ##############################################
            # WRITE YOUR CODE HERE
            # 여기서 self.features가 출력하는 모양을 flatten 했을 때
            # 노드 수를 계산하여 
            # 계산된 노드수->100으로 가는 완전 연결층을 구성하기
            torch.nn.Linear(backbone._fc.in_features, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(),
            torch.nn.Linear(512, 100),
            torch.nn.LogSoftmax(dim=-1)
            ##############################################
        )
    
    def forward(self, x):
        self.fmap = self.features.extract_features(x) # (N,3,300,300)->(N,1920,9,9)
        
        N = self.fmap.shape[0]
        x = self.avg_pool(self.fmap).reshape(N,-1) # (N,1920,9,9)->(N,1920,1,1)->(N,1920) 
        x = self.classifier(x) #(N,1920)->(N,100)

        return x