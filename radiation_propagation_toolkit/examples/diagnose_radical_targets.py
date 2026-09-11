"""Post-freeze auxiliary-target diagnostics; never changes model selection."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import torch
from qp_ode_simulator.radical_models import load,verify,Temporal,temporal_input,SEEDS


def main(root):
    verify(root);torch.set_num_threads(2);data,rows=load(root,True);fit,fitrows=load(root)
    tr=fitrows.role=='train';train_nu=fit['nuisance'][tr].astype(np.float64)
    nu_mean=train_nu.mean(0);nu_std=train_nu.std(0);variable=nu_std>1e-8
    field_mean=fit['privileged_mean'][tr].mean(0).reshape(-1)
    truth_field=data['privileged_mean'].reshape(len(rows),-1);truth_nu=data['nuisance']
    predictions={'constant_training_mean':(np.repeat(nu_mean[None],len(rows),axis=0),np.repeat(field_mean[None],len(rows),axis=0))}
    for arm in ['coinc_joint','coinc_aux']:
        nus=[];fields=[]
        for seed in SEEDS:
            b=joblib.load(root/f'{arm}_{seed}.joblib');model=Temporal();model.load_state_dict(b['state']);model.eval()
            x,_,_=temporal_input(data['time'],data['coinc'],arm,b['mean'],b['std'])
            with torch.no_grad():_,n,f=model(torch.from_numpy(x))
            nus.append(b['nu_scaler'].inverse_transform(n.numpy()));fields.append(b['field_scaler'].inverse_transform(f.numpy()))
        predictions[arm]=(np.mean(nus,axis=0),np.mean(fields,axis=0))
    records=[]
    for name,(nu,field) in predictions.items():
        for band in [0,1,2]:
            mask=(rows.strength_band==band).to_numpy()
            records.append(dict(model=name,strength_band=band,nuisance_standardized_rmse=float(np.sqrt(np.mean(((nu[mask][:,variable]-truth_nu[mask][:,variable])/nu_std[variable])**2))),
                field_probability_rmse=None if name=='coinc_joint' else float(np.sqrt(np.mean((field[mask]-truth_field[mask])**2)))))
    pd.DataFrame(records).to_csv(root/'auxiliary_diagnostic.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root',type=Path);main(p.parse_args().root)
