# Dataset sources and access

This project uses the EnCoV liver registration evaluation data and contour/model data released by Koo et al. Patient data and derived files are not redistributed in this public repository. Download the archives directly from their providers and follow the providers' terms.

- Evaluation dataset: [Dataset_v1p0.zip](https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/Dataset_v1p0.zip)
- Contours and simplified models: [Data_2017_Koo_etal_v1p0.zip](https://encov.ip.uca.fr/ab/code_and_datasets/datasets/llr_reg_evaluation_by_lus/Data_2017_Koo_etal_v1p0.zip)

SHA256 of the upstream archives:

```text
Dataset_v1p0.zip
EFE901F9DFAABF7AA9EF7F7A9EC25C88564C46934F9DB2C6ADC7DE7C550AC1E4

Data_2017_Koo_etal_v1p0.zip
113421C1D9C25911C98BABBD4FACF0C94D0F8E27563960B0058801C031D9679B
```

After downloading, verify the hashes and restore the Patient1 inputs to the expected `data/raw/` layout described by the upstream archive and project scripts. These links and hashes do not grant redistribution rights.
