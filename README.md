# 02456_DL_Project

## Data Strucure
```
data/
├── finetunning/
│   ├── CFRP_60_high/ 
│   └── CFRP_60_low/
└── pretraining/
```

## Data Preprocessing
- The dataset used in the `pretraining` folder is the following : [dataset link](https://opensource.silicon-austria.com/sabathiels/temperature-reconstruction/-/blob/main/datasets/onlySiliconSource_physical_units_fine/temperature_array_list.npy?ref_type=heads)
- `prepro_pretraining_data.py` shows how to open the pretraining dataset and use images.
It countains 8442 images with shape: (51, 51).