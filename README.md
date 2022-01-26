# Dynamic_mask_maker
make dynamic mask

### Patch-based 방법을 적용한 방식
<원본이미지>

<p align="left"><img src ="https://user-images.githubusercontent.com/63538314/151117709-53b09a47-db68-4aaf-9a13-39d9ad475564.png" width="400"></p>

<수정된 이미지>

<p align="left"><img src ="https://user-images.githubusercontent.com/63538314/151117940-4781b4fe-aba4-425d-8992-a36d90afdbb9.png" width="400"></p>

<참고 이미지>

<p align="left"><img src ="https://user-images.githubusercontent.com/63538314/151118130-9c45a355-8059-47e9-96bc-43b36ec073aa.jpg" width="400"></p>

dense-sift는 아래의 깃허브에서 얻어왔다.

https://github.com/Yangqing/dsift-python

### 비고 

구현간에 단점은 optical flow나 fundamental matrix가 부정확하게 구해지면 않좋은 이미지를 보여준다는점이다. 이에 대한 보정이 필요하다.

### 참고 문헌

G Kanojia, S Raman, "Simultaneous Detection and Removal of Dynamic Objects in Multi-view Images", CVPR 2020

Y. Jia and T. Darrell. ``Heavy-tailed Distances for Gradient Based Image Descriptors''. NIPS 2011
