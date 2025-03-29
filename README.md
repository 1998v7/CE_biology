# CE_biology

```bash
# install dependencies
pip install -r requirements.txt

# download and preprocess the dataset
python prepare_data.py

# run the training
python run_rnn.py --config ./configs/cath_arch.yaml
python run_rnn.py --config ./configs/cath_topo.yaml
```
