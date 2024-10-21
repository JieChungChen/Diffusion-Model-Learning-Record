# Diffusion Model學習紀錄  

記錄diffusion相關模型的知識，並用自己的習慣重新整理各種奇怪版本、可讀性糟糕的code，學到哪更新到哪。

## Denoising Diffusion Probabilistic Models (DDPM)  

附上我在hackmd上的筆記: [盡量白話解釋DDPM基礎](https://hackmd.io/@jackson29/rkpmHlK6C) 

## Reference  

這邊是我看到Github上面可讀性較高、適合初學者看的repo.

1. [DenoisingDiffusionProbabilityModel-ddpm-](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-)<br>應該是最適合初學者的看的，這個repo用Cifar10資料集去跑DDPM，絕大多數人的硬體都跑的了。它有完全沒加condition，以及用class當做condition，共兩種版本可以玩。但好像有一段在算posterior variance的部分看起來怪怪的，跟原論文有出入，不過我實際跑過後的結果影響不大。

2. 