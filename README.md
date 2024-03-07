# DistYOLO

## Description

DistYOLO is an implementation of the DistYOLO paper, designed with model scalability in mind to facilitate configuration with various inference environments. This project aims to provide a flexible and efficient implementation of DistYOLO using the YOLOv8 architecture.

## Installation

To use DistYOLO, simply clone the repository:

```
git clone https://github.com/yourusername/DistYOLO.git
```
DistYOLO relies on TensorFlow and Keras. You can install the required dependencies using pip:

```
pip install tensorflow keras
```

## Usage

As the model is not pretrained, users can import the YOLO class and train their own instance of it. Pretrained models may be included in the future.

## Contributions

Contributions, comments, issue reports, and pull requests are welcome. While the project is currently maintained solely by the author, contributions from the community are encouraged.

## License

DistYOLO is licensed under the Apache License, Version 2.0. This decision is influenced by the Apache License v2 of KerasCV, from which parts of this project were derived.

>Copyright 2024 AbdEl-Rahman El-Gharib
>
>Licensed under the Apache License, Version 2.0 (the "License");
>
>you may not use this file except in compliance with the License.
>
>You may obtain a copy of the License at
>
>>http://www.apache.org/licenses/LICENSE-2.0
>
>Unless required by applicable law or agreed to in writing, software
>
>distributed under the License is distributed on an "AS IS" BASIS,
>
>WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
>
>See the License for the specific language governing permissions and
>
>limitations under the License.

## Credits

The DistYOLO implementation is based on the DistYOLO paper with some modifications to comply with the YOLOv8. 
Parts of this code are derived from KerasCV with some modifications to comply with the DistYOLO architecture.

>@Article{app12031354,
>
>AUTHOR = {Vajgl, Marek and Hurtik, Petr and Nejezchleba, Tomáš},
>
>TITLE = {Dist-YOLO: Fast Object Detection with Distance Estimation},
>
>JOURNAL = {Applied Sciences},
>
>VOLUME = {12},
>
>YEAR = {2022},
>
>NUMBER = {3},
>
>ARTICLE-NUMBER = {1354},
>
>URL = {https://www.mdpi.com/2076-3417/12/3/1354},
>
>ISSN = {2076-3417},
>
>ABSTRACT = {We present a scheme of how YOLO can be improved in order to predict the absolute distance of objects using only information from a monocular camera. It is fully integrated into the original architecture by extending the prediction vectors, sharing the backbone’s weights with the bounding box regressor, and updating the original loss function by a part responsible for distance estimation. We designed two ways of handling the distance, class-agnostic and class-aware, proving class-agnostic creates smaller prediction vectors than class-aware and achieves better results. We demonstrate that the subtasks of object detection and distance measurement are in synergy, resulting in the increase of the precision of the original bounding box functionality. We show that using the KITTI dataset, the proposed scheme yields a mean relative error of 11% considering all eight classes and the distance range within [0, 150] m, which makes the solution highly competitive with existing approaches. Finally, we show that the inference speed is identical to the unmodified YOLO, 45 frames per second.},
>
>DOI = {10.3390/app12031354}
>
>}

>@misc{wood2022kerascv,
>
>title={KerasCV},
>
>author={Wood, Luke and Tan, Zhenyu and Stenbit, Ian and Bischof, Jonathan and Zhu, Scott and Chollet, Fran\c{c}ois and Sreepathihalli, Divyashree and Sampath, Ramesh and others},
>
>year={2022},
>
>howpublished={\url{https://github.com/keras-team/keras-cv}},
>
>}

## Contact

For questions, comments, or support, please use the comments section of the repository.

## Additional Information

Please note that this project is still under active development. Use with CAUTION!
