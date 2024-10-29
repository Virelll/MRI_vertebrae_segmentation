# Vertebra segmentation MRI

## Задача 

Реализовать и обучить модели  нейронных сетей для сегментирования и идентификации позвонков на сагиттальной проекции МРТ снимков позвоночника.  

![Image alt](https://github.com/Virelll/project1/blob/main/mri1.png)

_____

## Результат

Реализовано и обучены архитектуры UNet и ResUNet. Экспериментируя с гиперпараметрами удалось добиться наибольшего результата: 

UNet (Dice = 0.80894)

* Оптимизатор Adamax, начальный learning_rate = 0,001, epoches = 100, lr_scheduler(patience = 5, factor = 0.5)

ResUNet (Dice = 0.82904)

* Оптимизатор Adamax, начальный learning_rate = 0.001, epoches = 100, lr_scheduler(patience = 5, factor = 0.5) 
