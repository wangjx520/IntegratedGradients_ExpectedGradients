import numpy as np
import torch
from model import CNN
from utils import load_dataset
from attribution import integrated_gradients, expected_gradients

if __name__ == '__main__':
	
	device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

	trainset, testset = load_dataset()
	model = torch.load('model/cnn.pt').to(device)

	# Target Number : 0 ~ 9
	idx = 42
	target = 3
	n_iter = 10000
	
	# 利用test数据集作为计算ig值的基线
	baseline =testset.test_data  # [10000,28,28] mnist为灰度图
	baseline=baseline[:,np.newaxis,...].float().numpy() #[10000,1,28,28]
	 
	data = baseline[np.where(testset.test_labels == target)][idx:idx+1]

	ig_attr = integrated_gradients(model,data,target,n_iter,device=device,visualize=True)
	eg_attr,eg_var = expected_gradients(model,data,baseline,target,n_iter,device=device,visualize=True)
	
	
	 