# ERA-SESSION13 PyTorch Lightning &amp; Gradio

HF Link: https://huggingface.co/spaces/RaviNaik/ERA-Session12

### Tasks:
1. :heavy_check_mark: Move your S10 assignment to Lightning first and then to Spaces such that: 
 - (You have retrained your model on Lightning) 
 - You are using Gradio 
2. :heavy_check_mark: Your spaces app has these features: 
 - ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well 
 - ask whether he/she wants to view misclassified images, and how many 
 - allow users to upload new images, as well as provide 10 example images 
 - ask how many top classes are to be shown (make sure the user cannot enter more than 10) 
 - Add the full details on what your App is doing to Spaces README  
3. :heavy_check_mark: Then:  
 - Submit the Spaces App Link 
 - Submit the Spaces README link (Space must not have a training code) 
 - Submit the GitHub Link where Lightning Code can be found along with detailed README with log, loss function graphs, and 10 misclassified images
### Model Summary
```python
  | Name     | Type               | Params
------------------------------------------------
0 | prep     | Sequential         | 1.9 K 
1 | layer1   | Sequential         | 369 K 
2 | layer2   | Sequential         | 295 K 
3 | layer3   | Sequential         | 5.9 M 
4 | pool     | MaxPool2d          | 0     
5 | fc       | Linear             | 5.1 K 
6 | softmax  | Softmax            | 0     
7 | accuracy | MulticlassAccuracy | 0     
------------------------------------------------
6.6 M     Trainable params
0         Non-trainable params
6.6 M     Total params
26.292    Total estimated model params size (MB)
```

### LR Finder
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/bfbffdb8-614d-48c0-bff7-3acf71213b76)

### Accuracy
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/47d4b1da-2573-4022-b6b4-05d1f93d5757)

### Test Accuracy
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/a0959618-17c8-4ada-980f-6dc7ba76eb61)

### Tensorboard Plots
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/6337fc11-67c0-4039-a0b5-8238a0307eca)
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/78f2b7ad-b781-4009-8356-15a6ff512896)

![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/da4141b3-b13f-4390-9e24-519c8ac0f5b8)
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/d30ea859-6bb5-469c-9097-bc198595309f)

### Misclassified Images
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/75494f70-a533-4a70-8c11-ef4c63fce21b)

### GradCAM Images
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/4d2a75fa-3902-4839-a32a-bbfec4ef72ba)

### HuggingFace Interface
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/658535f0-a322-4b84-adce-840b0cd74807)
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/60f46957-d308-4538-9eef-1346c15aadf2)
![image](https://github.com/RaviNaik/ERA-SESSION12/assets/23289802/fc8d3d7c-cd2e-46d5-b599-b61b45845ee9)
