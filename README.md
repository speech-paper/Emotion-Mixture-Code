# Emotion-Mixture-Code

Speech demo: [Demo]().

## How to experience Emotion Mixture

### Preliminary work

- Download the [checkpoint](https://drive.google.com/file/d/1k4KINFlEr3c0pSsbmQEW0tWk1Rf01l1i/view?usp=drive_link) of the HiFi-GAN model and place it in the "checkpts" folder.
- Download the [checkpoint](https://drive.google.com/file/d/1_7zNeaTsugOn3aCKaSSe1WPlZzjfwS2m/view?usp=drive_link) of the MG-TTS model pre-trained on the LJSpeech dataset and place it in the "logs_new/ljs_pre_with_emo_new" folder.
- Download the task vectors of the fine-tuned model ([emotion_vector_sur](https://drive.google.com/file/d/1EsUkrzL89PAZ8tSPiQamXGHjOGtgfcOZ/view?usp=drive_link), [emotion_vector_ang](https://drive.google.com/file/d/1wpJo_dqAPggenx45lRj9E_qLUsBMqHsi/view?usp=drive_link)) and place them in the "emotion_vector/13/with_emo/" folder.

### Synthesize primary emotions

```python
inference_by_emo_vector.py --emo1 {sur or ang} --speaker_id 13
```

### Synthesize mixed emotions

```
inference_by_emo_vector.py --emo1 sur --emo2 ang --alpha 0.2 --speaker_id 13
```

Parameter Explanation:

- -- emo1: emotion A 
- -- emo2: emotion B
- -- alpha: The weight of the task vector for emotion A is alpha, while the weight of the task vector for emotion B is 1 - alpha.
- -- speaker_id: specific speaker

### Acknowledgement

+ [HiFi-GAN]([jik876/hifi-gan: HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://github.com/jik876/hifi-gan))
+ [Grad-TTS]([Speech-Backbones/Grad-TTS at main · huawei-noah/Speech-Backbones](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS))
+ [LJSpeech]([Speech-Backbones/Grad-TTS at main · huawei-noah/Speech-Backbones](https://github.com/huawei-noah/Speech-Backbones/tree/main/Grad-TTS))
+ [Emotion Speech Dataset (ESD)](https://github.com/HLTSingapore/Emotional-Speech-Data)

### License

Our code is released under MIT License. 