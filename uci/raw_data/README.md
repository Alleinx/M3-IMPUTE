# Datasets README.md

The datasets used in this work can obtains online or import from the suppmentary material we provide. All datasets should place inside the `uci/raw_data` folder.

- UCI (8 datasets): https://github.com/maxiaoba/GRAPE/tree/master/uci/raw_data
  - concrete, energy, housing, kin8nm, naval, power, wine, yacht
- Extra 17 datasets: include in suppmentary material archive
  - airfoil, blood, breast, diabetes, ionosphere, iris, wine-white, protein, spam, letter, abalone, ai4i, cmc, german, steel, libras, california-housing


Expected folder structure:

```
├── uci
│   ├── __init__.py
│   ├── raw_data
│   │   ├── abalone
│   │   ├── ai4i
│   │   ├── airfoil
│   │   ├── blood
│   │   ├── breast
│   │   ├── california-housing
│   │   ├── cmc
│   │   ├── concrete
│   │   ├── diabetes
│   │   ├── energy
│   │   ├── german
│   │   ├── housing
│   │   ├── ionosphere
│   │   ├── iris
│   │   ├── kin8nm
│   │   ├── letter
│   │   ├── libras
│   │   ├── naval
│   │   ├── power
│   │   ├── protein
│   │   ├── spam
│   │   ├── steel
│   │   ├── wine
│   │   ├── wine-white
│   │   └── yacht
│   ├── uci_data.py
│   └── uci_subparser.py
```

> Please make sure the `uci_data.py` and `uci_subparser.py` files is not overwrited with the one from `GRAPE`.