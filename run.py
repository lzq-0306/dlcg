import os
import numpy as np
targetList = np.random.randint(1,10000,1000)
nums = [7,17,27,37,47,57,77,77,87,97]
i = 25
img_path = r"results/20231102"
def LeNet():
    try:
        for n in range(2):
            # os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+
            #         str(targetList[n]) +' --input_cmp 0.01 --rec_cmp 0.01 --image_path '+ img_path)
            # os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+
            #         str(targetList[n]) +' --input_cmp 0.05 --rec_cmp 0.05 --image_path '+ img_path)
            # os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+
            #         str(targetList[n]) +' --input_cmp 0.1 --rec_cmp 0.1 --image_path '+ img_path)
            # os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+
            #         str(targetList[n]) +' --input_cmp 0.15 --rec_cmp 0.15 --image_path '+ img_path)
            # os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+
            #         str(targetList[n]) +' --input_cmp 0.2 --rec_cmp 0.2 --image_path '+ img_path)
            # os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+
            #         str(targetList[n]) +' --input_cmp 0.25 --rec_cmp 0.25 --image_path '+ img_path)
            # os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+
            #         str(targetList[n]) +' --input_cmp 0.3 --rec_cmp 0.3 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model LeNetZhu --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim zhu  --restarts 1 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.9 --rec_cmp 0.9 --tv 0 --image_path '+ img_path)
        # for i in range(100):
        #     os.system('python reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+ str(i+1) +' --input_cmp 0.2 --rec_cmp 0.2 --image_path '+ r"C:\Users\Smoker\OneDrive\invertinggradients-master\images")
    except KeyboardInterrupt:
        pass



def ResNet():
    try:
        for n in range(500):
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.01 --rec_cmp 0.01 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.05 --rec_cmp 0.05 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.1 --rec_cmp 0.1 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.15 --rec_cmp 0.15 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.2 --rec_cmp 0.2 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.25 --rec_cmp 0.25 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.3 --rec_cmp 0.3 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.35 --rec_cmp 0.35 --image_path '+ img_path)
        # for i in range(100):
        #     os.system('python reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+ str(i+1) +' --input_cmp 0.2 --rec_cmp 0.2 --image_path '+ r"C:\Users\Smoker\OneDrive\invertinggradients-master\images")
    except KeyboardInterrupt:
        pass

def ImageNet():
    try:
        for n in range(1000):
          #  os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
          #          str(targetList[n]) +' --input_cmp 0.01 --rec_cmp 0.01 --image_path '+ img_path)
          #  os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
          #          str(targetList[n]) +' --input_cmp 0.05 --rec_cmp 0.05 --image_path '+ img_path)
          #  os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
          #          str(targetList[n]) +' --input_cmp 0.1 --rec_cmp 0.1 --image_path '+ img_path)
          #  os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
          #          str(targetList[n]) +' --input_cmp 0.15 --rec_cmp 0.15 --image_path '+ img_path)
          #  os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
          #          str(targetList[n]) +' --input_cmp 0.2 --rec_cmp 0.2 --image_path '+ img_path)
          #  os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
          #          str(targetList[n]) +' --input_cmp 0.25 --rec_cmp 0.25 --image_path '+ img_path)
            os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
                    str(targetList[n]) +' --input_cmp 0.3 --rec_cmp 0.3 --image_path '+ img_path)
          #  os.system('python3 reconstruct_image.py --model ResNet18 --dataset ImageNet --trained_model --cost_fn sim --indices def --optim ours  --restarts 3 --save_image --target_id '+
          #          str(targetList[n]) +' --input_cmp 0.35 --rec_cmp 0.35 --image_path '+ img_path)
        # for i in range(100):
        #     os.system('python reconstruct_image.py --model ResNet20-4 --dataset CIFAR10 --trained_model --cost_fn sim --indices def --optim ours  --restarts 1 --save_image --target_id '+ str(i+1) +' --input_cmp 0.2 --rec_cmp 0.2 --image_path '+ r"C:\Users\Smoker\OneDrive\invertinggradients-master\images")
    except KeyboardInterrupt:
        return
if __name__ == '__main__':
    ImageNet()
#     LeNet()
    # ResNet()
