# Emotion-Classficication

## Train
ResNet18

lr = 0.01

batch_size = 32

momentum = 0.9

weight_decay = 1e-4

epoch = 35


## DATA  

    - (1) training data : 21,529files

                   -angry : 2996 files

                   -disgusted : 327 files

                   -fearful : 3072 files

                   -happy : 5411 files

                   -neutral : 3723 files

                   -sad : 3622 files

                   -surprised : 2378 files

    - (2) test data : 7,178files

                   -angry : 958 files

                   -disgusted : 111 files

                   -fearful : 1024 files

                   -happy : 1774 files

                   -neutral : 1233 files

                   -sad : 1247 files

                   -surprised : 831 files

    - (3) validation data : 7180files

                   -angry : 999 files

                   -disgusted : 109 files

                   -fearful : 1025 files

                   -happy : 1804 files

                   -neutral : 1242 files

                   -sad : 1208 files

                   -surprised : 793 files

- data 특징
    - 실제 인물 사진이 아닌 그림도 존재
    - 안경 쓴  인물 이미지도 있음
    - disgusted, angry, fearful의 이미지 별 특징이 비슷함
- img.shape = (48, 48, 3)

DB : https://www.kaggle.com/ananthu017/emotion-detection-fer

## Result

best_acc = 66.04

![123](https://user-images.githubusercontent.com/85150131/129670337-e316acda-dfe7-48b7-9412-f23a4eab452e.png)
