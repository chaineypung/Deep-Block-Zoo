# Deep Block Zoo repository
[![Stable release](https://img.shields.io/badge/version-2022.0.0-green.svg)](https://github.com/openvinotoolkit/open_model_zoo/releases/tag/2022.0.0)
[![Gitter chat](https://badges.gitter.im/gitterHQ/gitter.png)](https://gitter.im/open_model_zoo/community)
[![Apache License Version 2.0](https://img.shields.io/badge/license-Apache_2.0-green.svg)](LICENSE)

<p align="center">
<img src="./picture.png" alt="drawing" width="100%" height="30%"/>
    <h4 align="center"> </h4>
</p>

Some plug-and-play network modules.

These modules are separated from the classical network and they can help you improve the upper limit of your model.

| Block Name | Input Shape | Output Shape | Parameters(M) |
|:----:|:----:|:----:|:----:|
| NAF | 4, 16, 256, 256 | 4, 16, 256, 256 | 0.00232 |
| ASPP | 4, 16, 256, 256 | 4, 16, 256, 256 | 0.00145 |
| Non-local | 4, 16, 64, 64 | 4, 16, 64, 64 | 0.00051 |
| SE | 4, 16, 256, 256 | 4, 16, 256, 256 | 0.00003 |
| CBAM |4, 16, 256, 256 | 4, 16, 256, 256 | 0.00030 |
| DCN | 4, 16, 256, 256 | 4, 16, 256, 256 | 0.00491 |
| ASFF |4, 512, 64, 64; 4, 256, 128, 128; 4, 256, 256, 256 | 4, 256, 256, 256 | 0.73445 |
| RFB | 4, 16, 256, 256 | 4, 16, 256, 256 | 0.00137 |
| Ghost | 4, 16, 256, 256 | 4, 16, 256, 256 | 0.00023 |

## License
Deep Block Zoo is licensed under [Apache License Version 2.0](LICENSE).
