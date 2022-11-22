import torch
import matplotlib.pyplot as plt
import numpy as np
import copy

def gradient(model,inputs):
	model.zero_grad()

	X=[x.requires_grad_() for x in inputs]
	outputs=model(*X)[:,2]
	selected=[val for val in outputs]

	grads=[torch.autograd.grad(selected,x,retain_graph=True)[0].cpu().numpy()
	  for idx,x in enumerate(X)]

	return grads

def vis(data,ig):

	plt.figure(figsize=(15, 8))
	plt.subplot(1, 2, 1)
	plt.imshow(data.reshape(28, 28, 1), cmap='bone_r')
	plt.grid()
	plt.axis('off')
	plt.subplot(1, 2, 2)
	plt.imshow(ig.reshape(28, 28, 1), cmap='bone_r')
	plt.grid()
	plt.axis('off')
	plt.show()
	
def integrated_gradients(model, inputs, target_num, nsamples, device, visualize=True):
	assert isinstance(inputs, np.ndarray), "data should be np.ndarray type"
	img_data=copy.copy(inputs)
	assert inputs.ndim == 4, "(n_batch, n_feat, h, w)"

	inputs   = torch.from_numpy(inputs).float().to(device)
	baseline = torch.from_numpy(np.random.random(size=(inputs.shape))).float().to(device)  # 生成路径演化的末端值

	alpha = np.linspace(0, 1, nsamples)
	alpha = torch.from_numpy(alpha).float().to(device)
	alpha = alpha.view(nsamples, *tuple(np.ones(baseline[0].ndim, dtype='int')))  # [10000,1,1,1]

	attributions = []

	sampled_datas=baseline+alpha*(inputs-baseline)
	attribution=torch.zeros(sampled_datas.shape).to(device)

	for i in range(nsamples):

		sampled_data=sampled_datas[i:i+1]
		sampled_data.requires_grad_(True)
		outputs=model(sampled_data)[:,target_num]
		attribution[i]=torch.autograd.grad(outputs,sampled_data)[0]

	integrated=attribution.sum(axis=0)/nsamples
	ig=(inputs-baseline)*integrated
	ig=ig.detach().cpu().numpy().squeeze()

	if visualize:
		vis(img_data,ig)
	return ig



def expected_gradients(model, inputs, baseline, target_num, nsamples, device, visualize=True):
	
	assert isinstance(inputs, np.ndarray), "data should be np.ndarray type"
	img_data=copy.copy(inputs)
	assert inputs.ndim == 4, "(n_batch, n_feat, h, w)"

	replace  = baseline.shape[0]<nsamples
	sample_idx=np.random.choice(baseline.shape[0],size=(nsamples),replace=replace)
	sampled_baseline=baseline[sample_idx]
	#sampled_baseline = torch.from_numpy(baseline[sample_idx]).float().to(device)  # 生成路径演化的末端值


	attributions = []

	sampled_datas=np.zeros((nsamples,)+inputs.shape[1:])
	sampled_delta=np.zeros((nsamples,)+inputs.shape[1:])

	attribution=torch.zeros(sampled_datas.shape).to(device)

	for i in range(nsamples):

		data=copy.copy(inputs)
		idx=np.random.choice(baseline.shape[0])
		alpha=np.random.uniform()


		sampled_datas[i]=alpha*data+(1-alpha)*sampled_baseline[i]
		sampled_delta[i]=data-baseline[idx]

	grads=[]
	
	batch_size=199
	for i in range(0,nsamples,batch_size):
		batch_inputs=sampled_datas[i:min(i+batch_size,nsamples)]
		batch_inputs=torch.from_numpy(batch_inputs).float().to(device=device)

		grads.append(gradient(model,[batch_inputs]))

	grad =np.hstack(grads).squeeze(0)
	samples=grad*sampled_delta
	eg_values=samples.mean(0).squeeze(0)
	eg_var=samples.var()/np.sqrt(nsamples)
	if visualize:
		vis(img_data,eg_values)
	return eg_values,eg_var

