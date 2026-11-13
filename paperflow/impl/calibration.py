"""Calibration matrix loader and Cholesky whitening for paper K-SVD."""
import os
import pickle
import torch

class Calib:
    @staticmethod
    def save(path, data):
        with open(path, 'wb') as f:
            pickle.dump(data, f)

    @staticmethod
    def load(path):
        with open(path, 'rb') as f:
            return pickle.load(f)

    @staticmethod
    def get_calib_data(group, name, save_path=None):
        assert save_path is not None
        # calibration data for up and gate is the same
        if name == 'mlp.gate_proj':
            name = 'mlp.up_proj'
        if name == "self_attn.q_proj" or name == "self_attn.v_proj":
            name = "self_attn.k_proj"
        data = None
        file_path = os.path.join(save_path, name)
        for item in group:
            file_name = os.path.join(file_path, "{}.pkl".format(item))
            if os.path.exists(file_name):
                tmp_data = Calib.load(file_name)
                if data is None:
                    data = tmp_data
                else:
                    data += tmp_data
            else:
                raise FileNotFoundError(
                    "{} not found. You should run build_calibration_dataset first!".format(file_name))
        return data

    @staticmethod
    def get_s_inv_s(group, name, model_type, calib_path=None):
        data = Calib.get_calib_data(group, name, calib_path).double()
        # The following code is from https://github.com/AIoT-MLSys-Lab/SVD-LLM
        try:
            scaling_diag_matrix = torch.linalg.cholesky(data).T
        except Exception as e:
            print("Warning: eigen scaling_diag_matrix is not positive!")
            eigenvalues = torch.linalg.eigvalsh(data)
            data += (- eigenvalues[0] + 7e-6) * torch.eye(data.shape[0]).to(data.device)
            scaling_diag_matrix = torch.linalg.cholesky(data).T
            eigenvalues = None
            del eigenvalues
        invs = torch.linalg.inv(scaling_diag_matrix)
        return scaling_diag_matrix, invs

