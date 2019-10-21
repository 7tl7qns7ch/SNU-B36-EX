# SNU-B36-EX
**SNU-B36-EX** is an inter-floor noise dataset which is extended version of **SNU-B36-50** collected in building no.36 at Seoul National University, Seoul, Korea. It was recorded with a smartphone (Samgsung Galaxy S6) with sampling rate 44,100Hz. The data set is available at [SNU-B36-EX](https://drive.google.com/open?id=1uzvEywFV0KmhOq0xWewuNqMPXtE1qAyH). The name of each folder indicate class number (see Train/test split for Zero-Shot settings below).

All setting and procedure are same as the previous works, but it has the additional noise data by setting the horizontal source points in 1m increments from 0 to 12m. Thus, there are 8,450 audio files in the data set.

The dataset consists of 5 types: 

'MB' : a medicine ball on the floor,

'HD' : dropping a hammer on the floor,

'HH' : hitting with a hammer on the floor,

'CD' : dragging a chair on the floor,

'VC' : operating a vacuum cleaner,

![](https://github.com/7tl7qns7ch/SNU-B36-EX/blob/master/figures/noise_type.png)

and 39 positions: 

3F0m, 3F1m, 3F2m, 3F3m, 3F4m, 3F5m, 3F6m, 3F7m, 3F8m, 3F9m, 3F10m, 3F11m, 3F12m,

2F0m, 2F1m, 2F2m, 2F3m, 2F4m, 2F5m, 2F6m, 2F7m, 2F8m, 2F9m, 2F10m, 2F11m, 2F12m,

1F0m, 1F1m, 1F2m, 1F3m, 1F4m, 1F5m, 1F6m, 1F7m, 1F8m, 1F9m, 1F10m, 1F11m, 1F12m.

![](https://github.com/7tl7qns7ch/SNU-B36-EX/blob/master/figures/noise_position.png)



**Train/test set split for Zero-Shot settings**

Train set is colored by black, test set is colored by red, and numbers colored by green are empty set.
![](https://github.com/7tl7qns7ch/SNU-B36-EX/blob/master/figures/data_classes_number.PNG)
