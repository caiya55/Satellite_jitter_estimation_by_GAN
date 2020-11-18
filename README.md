# Remote sensing image restoration based on generative adversarial network (GAN)
High-resolution satellite images (HRSIs) obtained from onboard satellite linear array cameras suffer from geometric disturbance in the presence of attitude jitter. Therefore,
detection and compensation of satellite attitude jitter are crucial to reduce the geo-positioning errors and improve the geometric  accuracy of HRSIs. In this work, a generative adversarial network (GAN) architecture is proposed to automatically learn and correct the deformed scene features from a single remote sensing image. In the proposed GAN, a convolutional neural network (CNN) is designed to discriminate the inputs and another CNN is used to generate so-called fake inputs. In order to explore
the usefulness and effectiveness of GAN for jitter detection, the proposed GAN are trained on part of PatternNet dataset and tested on three popular remote sensing datasets along with deformed Yaogan-26 satellite image. Several experiments show that the proposed models provide competitive results compared to other methods. The proposed GAN reveals the huge potential of GAN-based methods for the analysis of attitude jitter from remote sensing images.


![image](https://github.com/lexsaints/powershell/blob/master/IMG/ps2.png)
Fig. 1: Overview of the proposed RestoreGAN
