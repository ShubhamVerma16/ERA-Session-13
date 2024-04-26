# ERA-SESSION13 PyTorch Lightning &amp; Gradio

HF Link: https://huggingface.co/spaces/SV12/ERA_Session13

### Tasks:
- [x] Move your S11 assignment to Lightning first and then to Spaces such that: 
  - (You have retrained your model on Lightning) 
  - You are using Gradio 
- [x] Your spaces app has these features: 
  - ask the user whether he/she wants to see GradCAM images and how many, and from which layer, allow opacity change as well 
  - ask whether he/she wants to view misclassified images, and how many 
  - allow users to upload new images, as well as provide 10 example images 
  - ask how many top classes are to be shown (make sure the user cannot enter more than 10) 
  - Add the full details on what your App is doing to Spaces README  

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
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/9cecc3eb-0c47-4c6b-892e-fbd12ba30a9e)

### Test Accuracy
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/4eb2d59f-85c8-4c75-bd94-be7af50d3e70)

### Tensorboard Plots
#### Train Acc
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/9bd658e8-23d0-46d8-ad3a-9cd998753988)

#### Train Loss
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/0bcd1766-a30a-4dd3-b1de-82c5354694d3)

#### Validation Acc
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/ffe7a8c6-3187-43c4-8a35-6e50898484e7)

#### Validation Loss
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/95d99291-32ee-404c-b797-d2e880521e4d)


### Misclassified Images
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/b329a39b-18be-477b-9af4-4f67c41c4a94)

### GradCAM Images
![image](https://github.com/ShubhamVerma16/ERA-Session-13/assets/46774613/ba2c5cc1-7bc5-40ab-b83c-0115d6edb36e)

### HuggingFace Interface

