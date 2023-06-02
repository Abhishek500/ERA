import torch
import matplotlib.pyplot as plt

def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()

	

def plot_images(train_load):
	batch_data, batch_label = next(iter(train_load)) 
	fig = plt.figure()
		
	for i in range(12):
	  plt.subplot(3,4,i+1)
	  plt.tight_layout()
	  plt.imshow(batch_data[i].squeeze(0), cmap='gray')
	  plt.title(batch_label[i].item())
	  plt.xticks([])
	  plt.yticks([])
