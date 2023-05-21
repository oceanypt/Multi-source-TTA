## Multi-Source Test-Time Adaptation as Dueling Bandits for Extractive Question Answering

This repository contains the code for our paper [**Multi-Source Test-Time Adaptation as Dueling Bandits for Extractive Question Answering**]() at ACL 2023.

### Setup
You should run the following script to install the dependencies first.


```bash
pip install --user .
pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
```
(To be notated, the results in our paper were obtained by training on the GPU of NVIDIA A100.)


### Reproduce

To obtain the results of Bandit, run as follows:
```bash
bash scrips/multi_arm_ucb.bandit.sh
bash scrips/multi_arm_ucb.spanbert.bandit.sh
```

To obtain the results of UCB and Co-UCB, run as follows:
```bash
bash scrips/multi_arm_ucb.sh
bash scrips/multi_arm_ucb.spanbert.sh
```





