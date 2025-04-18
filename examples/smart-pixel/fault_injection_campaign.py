import os
import yaml
import time
import pickle
import argparse
import numpy as np
import fkeras as fk
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from datetime import datetime
from fkeras.metrics.hessian import HessianMetrics

import models


###################################################################################################

def load_data(base_dir="./data/ds8_only", local_id=0):
    train_data = "{}/dec6_ds8_quant/QuantizedInputTrainSetLocal{}.csv".format(
        base_dir, local_id
    )
    train_label = "{}/dec6_ds8_quant/TrainSetLabelLocal{}.csv".format(
        base_dir, local_id
    )
    test_data = "{}/dec6_ds8_quant/QuantizedInputTestSetLocal{}.csv".format(
        base_dir, local_id
    )
    test_label = "{}/dec6_ds8_quant/TestSetLabelLocal{}.csv".format(base_dir, local_id)


    df1 = pd.read_csv(train_data)
    df2 = pd.read_csv(train_label)
    df3 = pd.read_csv(test_data)
    df4 = pd.read_csv(test_label)

    X_train = df1.values
    X_test = df3.values
    y_train = df2.values
    y_test = df4.values

    print("Training set shape         :", X_train.shape)
    print("Training set shape (labels):", y_train.shape)
    print("Test set shape             :", X_test.shape)
    print("Test set shape (labels)    :", y_test.shape)

    return X_train, y_train, X_test, y_test

###################################################################################################

def load_model(config, pretrained_model=None):
    build_model = getattr(models, config["model"]["name"])
    if "fkeras" in config["model"]["name"]:
        model = build_model(
            config["model"]["input_shape"], 
            dense_width=config["model"]["dense_width"],
            logit_total_bits=config["model"]["logit_total_bits"],
            logit_int_bits=config["model"]["logit_int_bits"],
            activation_total_bits=config["model"]["activation_total_bits"],
            activation_int_bits=config["model"]["activation_int_bits"],
        )
    else: # Float
         model = build_model(
            config["model"]["input_shape"], 
            dense_width=config["model"]["dense_width"],
         )
    # Load pretrained model
    if pretrained_model:
        model.load_weights(pretrained_model)
    return model


###################################################################################################

def gen_smart_pix_0mispredicts_dataset(model, X_test, y_test):
    

    y_pred_og = model.predict(X_test, verbose=0)

    #Only keep x if model did classify it correctly
    x_test_pred_correct = list()
    y_pred_og_correct = list()
    for i, class_gtruth in enumerate(y_test):
        class_pred = np.argmax(y_pred_og[i])
        if class_pred == class_gtruth[0]:
            x_test_pred_correct.append(X_test[i])
            y_pred_og_correct.append(y_pred_og[i])

    x_test_pred_correct = np.array(x_test_pred_correct[:])
    y_pred_og_correct = np.array(y_pred_og_correct[:])

    print(f"x_test_pred_correct / x_test: {len(x_test_pred_correct)} / {len(X_test)}")
    print(f"Percent x_test_pred_correct: {len(x_test_pred_correct) / len(X_test)}")

    return x_test_pred_correct, y_pred_og_correct


###################################################################################################

def my_eval_metric_00(y_test, y_pred):
    assert y_test.shape == y_pred.shape
    num_mispredicts = 0
    for i in range(y_test.shape[0]):
        if np.argmax(y_test[i]) != np.argmax(y_pred[i]):
            num_mispredicts += 1
 
    return num_mispredicts

###################################################################################################

def my_eval_metric_01(y_test, y_pred):
    assert y_test.shape == y_pred.shape
    max_dist = 0
    for i in range(y_test.shape[0]):
        curr_dist = np.linalg.norm(y_test[i] - y_pred[i])
        if curr_dist > max_dist:
            max_dist = curr_dist
 
    return max_dist

###################################################################################################

def my_alert_func_00(metric_val_om, metric_val_fm):
    """
    This function returns a boolean value that will be used by the FIC loop
    to determine whether an alert should be printed to the FIC info log file.
    During the FIC loop, this function will be applied to every faulty model.
    
    This alert will return True if the evaluation metric value for the faulty
    model differs from that of the original model. Otherwise, it returns False.
    """

    return metric_val_fm != metric_val_om

###################################################################################################

def my_alert_func_01(metric_val_om, metric_val_fm):
    """
    This function returns a boolean value that will be used by the FIC loop
    to determine whether an alert should be printed to the FIC info log file.
    During the FIC loop, this function will be applied to every faulty model.
    
    This alert will return True if the evaluation metric value for the faulty
    model is strictly greater than that of the original model. Otherwise, it
    returns False.
    """

    return metric_val_fm > metric_val_om

###################################################################################################

def fic(fic_config):
    """
    This function performs a fault injection campaign (FIC) on a provided model
    """

    __tool_name__ = "fic"
    __version__ = "0.0.0"
    __date__ = "11-14-2024"

    #STEP: Print to tool and date info to new fic log file
    fic_t_start = time.time()
    fic_log_suffix = datetime.fromtimestamp(int(fic_t_start)).strftime('YMD_%Y_%m_%d__HMS_%H_%M_%S')
    fic_range_str = f"{fic_config['fic_range'][0]}_{fic_config['fic_range'][1]}_{fic_config['fic_range'][2]}"

    fic_pickle_fp = os.path.join(fic_config['fic_output_dir'], f"fic__for__{fic_config['model_id']}__range_{fic_range_str}__on__{fic_log_suffix}.pkl"        )
    fic_log_r_fp  = os.path.join(fic_config['fic_output_dir'], f"fic__for__{fic_config['model_id']}__range_{fic_range_str}__on__{fic_log_suffix}__results.log")
    fic_log_i_fp  = os.path.join(fic_config['fic_output_dir'], f"fic__for__{fic_config['model_id']}__range_{fic_range_str}__on__{fic_log_suffix}__info.log"   )

    fic_log      = open(fic_log_r_fp, "w")
    fic_log_info = open(fic_log_i_fp, "w")
    print(f"[Tool: {__tool_name__} | Version: {__version__} | Last Updated (YYYY-MM-DD): {__date__} | Current Date/Time: {datetime.now()}]", file=fic_log)
    print(f"[Tool: {__tool_name__} | Version: {__version__} | Last Updated (YYYY-MM-DD): {__date__} | Current Date/Time: {datetime.now()}]", file=fic_log_info)


    #STEP: Output info message to default console and info log file
    file_info_message  = f"[fic] The current fault injection campaign has (or will) create these files:\n"
    file_info_message += f"[fic] |-- (      Created) Log File (Results)    : {fic_log_r_fp }\n"
    file_info_message += f"[fic] |-- (      Created) Log File (Info/Alerts): {fic_log_i_fp }\n"
    file_info_message += f"[fic] |-- (To Be Created) Final Results/Metadata: {fic_pickle_fp}\n"
    print(file_info_message, file=fic_log_info)
    print(file_info_message)


    #STEP: Create list to store metric results (and more)
    metric = list()
    alerts = list()


    #STEP: Instantiate the FKeras model to be used
    fmodel = fk.fmodel.FModelAlt(fic_config["model"], incl_biases=True)
    print(f"[fic] FModel.layer_bit_ranges: {fmodel.layer_bit_ranges}", file=fic_log_info)
    print(f"[fic] FModel.num_model_param_bits: {fmodel.num_model_param_bits}", file=fic_log_info)


    #STEP: Compute and save the provided metric for the original/unaltered model
    y_pred = fmodel.model.predict(fic_config["X_test"], verbose=0)
    metric.append( (-1, fic_config["eval_metric_func"](y_pred, y_pred)) )
    print(f"[fic] Original Model (OM) Metric Value: {metric[0][1]}\n", file=fic_log_info)
    print(f"[fic] Note: OM = Original Model | FM = Faulty Model\n", file=fic_log_info)
    
    #STEP: Add latest metric value to result log
    print(metric[-1], file=fic_log)
    
    # print(f"fic_range = {fic_config['fic_range']}")
    # range_list = range(*fic_config['fic_range'])
    # print(f"Range list: {range_list}")

    # Convert Hessian-ranked parameters into bit index lists
    wbi_lists = convert_params_into_bit_lists(
        fic_config["hess_ranking"],
        layer_precision_info=fic_config["layer_precision_info"],
        bits_per_weight=fic_config["bit_width"],
    )
    wbi_list_delta_metrics = [[] for _ in range(len(wbi_lists))]

    num_bits_flipped = 0
    #STEP: Update fic_config
    fic_config["y_pred"] = y_pred
    fic_config["fmodel"] = fmodel
    fic_config["metric"] = metric
    fic_config["fic_log"] = fic_log
    fic_config["fic_log_info"] = fic_log_info
    fic_config["alerts"] = alerts

    curr_num_bits_flipped = probe_bit_lists(fic_config, wbi_lists, wbi_list_delta_metrics)
    num_bits_flipped += curr_num_bits_flipped

    while any(len(wbi_list) > 0 for wbi_list in wbi_lists):
        sensitivity_pointer = compute_sensitivity_pointer(
            wbi_lists, wbi_list_delta_metrics, fic_config["last_k"],
        )
        bit_idx = wbi_lists[sensitivity_pointer][0]
        delta_metric = flip_bit(fic_config, bit_idx)
        wbi_list_delta_metrics[sensitivity_pointer].append(delta_metric)
        # Limit wbi_list_delta_metrics to last_k_measurements
        if len(wbi_list_delta_metrics[sensitivity_pointer]) > fic_config["last_k"]:
            wbi_list_delta_metrics[sensitivity_pointer].pop(0)
        # Remove bit from list
        wbi_lists[sensitivity_pointer].pop(0)
        num_bits_flipped += 1

    assert num_bits_flipped == fmodel.num_model_param_bits

    #STEP: Print final tool and date/time info to fic log files
    fic_t_end = time.time()
    fic_t_delta  = datetime.fromtimestamp(fic_t_end)
    fic_t_delta -= datetime.fromtimestamp(fic_t_start)
    print(f"[Tool: {__tool_name__} | Version: {__version__} | Last Updated (YYYY-MM-DD): {__date__} | Current Date/Time: {datetime.now()} | Elapsed Time: {fic_t_delta}]", file=fic_log)
    print(f"[Tool: {__tool_name__} | Version: {__version__} | Last Updated (YYYY-MM-DD): {__date__} | Current Date/Time: {datetime.now()} | Elapsed Time: {fic_t_delta}]", file=fic_log_info)
    

    #STEP: Close log files
    fic_log.close()
    fic_log_info.close()


    #STEP: Save fic data as pickle file
    fic_data = {
        # "fic_config"                 : fic_config,
        "arg --config"               : fic_config["load_model_tuple"][0],
        "arg --pretrained-model"     : fic_config["load_model_tuple"][1],
        "arg --model-id"             : fic_config["model_id"],
        "FModel.model.to_json()"     : str(fmodel.model.to_json()),
        "fic_eval_metric_func_name"  : fic_config["eval_metric_func"].__name__,
        "fic_alert_func_name"        : fic_config["alert_func"].__name__,
        "fic_range"                  : fic_config["fic_range"],
        "fic_time_start"             : str(datetime.fromtimestamp(fic_t_start)),
        "fic_time_end"               : str(datetime.fromtimestamp(fic_t_end)),
        "fic_time_elapsed"           : str(fic_t_delta),
        "list of (bi, metric_value)" : metric,
        "list of (bi_causing_alert)" : alerts,
    }
    with open(fic_pickle_fp, "wb") as fo:
        pickle.dump(fic_data, fo)


###################################################################################################


def flip_bit(fic_config, bit_i):
    """
    Flip a bit in the model and compute the evaluation metric and log.
    Return metric of faulty model.
    """
    fmodel = fic_config["fmodel"]
    metric = fic_config["metric"]
    fic_log = fic_config["fic_log"]
    fic_log_info = fic_config["fic_log_info"]

    #STEP: Flip the desired bit(s) in the "original" model to make it "faulty"
    fmodel.explicitly_flip_bits([bit_i])

    #STEP: Compute and save the provided metric for the faulty model
    y_pred_fault = fmodel.model.predict(fic_config["X_test"], verbose=0)
    metric_value = fic_config["eval_metric_func"](fic_config["y_pred"], y_pred_fault)
    metric.append((bit_i, metric_value))

    #STEP: Add latest metric value to result log
    print(metric[-1], file=fic_log)
    fic_log.flush()

    #STEP: Add alert to info log if alert conditions satisfied
    if fic_config["alert_func"](metric[0][1], metric_value):
        alert_str  = f"[fic] ALERT: {fic_config['alert_func'].__name__} returned {True} on current OM, FM metric values\n"
        alert_str += f"[fic]   |  : OM Metric Value  : {metric[0][1]}\n"
        alert_str += f"[fic]   |  : FM Metric Value  : {metric_value}\n"
        alert_str += f"[fic]  END : Bit Flip Location: {bit_i} (global-bit-index)\n"
        print(alert_str, file=fic_log_info)
        fic_config["alerts"].append(bit_i)
        fic_log_info.flush()

    fmodel.explicitly_reset_bits([bit_i])
    return metric_value


###################################################################################################

def probe_bit_lists(fic_config, wbi_lists, wbi_list_delta_metrics):
    """
    PrioriFI helper function

    Flip first bit in each non-empty list. Return number of bits flipped.
    """
    num_bits_flipped = 0
    # Flip first bit in each list
    for i, wbi_list in enumerate(wbi_lists):
        if len(wbi_list) == 0:
            continue
        bit_idx = wbi_list[0]
        # print(f"Flipping bit {bit_idx}")
        delta_metric = flip_bit(fic_config, bit_idx)
        wbi_list_delta_metrics[i].append(delta_metric)
        # Remove bit from list
        wbi_list.pop(0)
        num_bits_flipped += 1
    return num_bits_flipped

###################################################################################################

def compute_sensitivity_pointer(wbi_lists, wbi_list_delta_metrics, last_k_measurements):
    """
    PrioriFI helper function

    Given a list of lists of bit indices, compute the sensitivity pointer of 
    which list to flip next.
    """
    # Compute median of last k delta metrics for each list
    last_delta_metrics = []
    for i in range(len(wbi_lists)):
        if len(wbi_lists[i]) >= last_k_measurements:
            # Take the median of the last k delta metric measurements
            last_delta_metrics.append(np.median(wbi_list_delta_metrics[i][-last_k_measurements:]))
        else: # Take the last delta metric (or average?)
            last_delta_metrics.append(np.mean(wbi_list_delta_metrics[i]))
    # last_delta_metrics = [wbi_list_delta_metrics[i][-1] for i in range(len(wbi_lists))]
    # print(f"last_delta_metrics = {last_delta_metrics}")
    # Take negative to sort in descending order
    sensivity_metric_argsort = np.argsort(-np.array(last_delta_metrics)) 
    # print(f"sensivity_metric_argsort = {sensivity_metric_argsort}")
    for i in range(len(sensivity_metric_argsort)):
        if len(wbi_lists[sensivity_metric_argsort[i]]) > 0:
            sensitivity_pointer = sensivity_metric_argsort[i]
            # print(f"Found sensitivity_pointer: {sensitivity_pointer}")
            break
    return sensitivity_pointer

###################################################################################################


def convert_params_into_bit_lists(param_ranking, layer_precision_info=None, bits_per_weight=None):
    """
    PrioriFI helper function

    Given a list of parameters, return a list of lists where each list contains
    the bit indices of the bits in the parameter.
    List 0 contains the MSBs, List 1 contains the MSB-1s, etc.
    """
    if type(layer_precision_info) == list:
        layer_info_head = 0
        num_of_params_seen = 0
        last_bit_idx = -1
        param_to_bit_indices_dict = dict()
        # Build a dictionary such that (key, val) = param_index, ([bit_indices])
        for wi in range(len(param_ranking)):
            bit_indices_associated_with_param = []
            bit_width = layer_precision_info[layer_info_head][1]
            for wbi in range(bit_width):
                last_bit_idx += 1
                bit_indices_associated_with_param.append((last_bit_idx, wbi))

            param_to_bit_indices_dict[wi] = bit_indices_associated_with_param

            num_of_params_seen += 1
            if num_of_params_seen == layer_precision_info[layer_info_head][0]:
                layer_info_head += 1
                num_of_params_seen = 0

        # Add the bit indices into the appropriate list based on
        # the parameter ranking
        max_bit_width = max(layer_precision_info, key=lambda x: x[1])[1]
        sorted_msb_lsb_lists = [[] for _ in range(max_bit_width)]
        for param_idx in param_ranking:
            for bit_idx, wbi in param_to_bit_indices_dict[param_idx]:
                sorted_msb_lsb_lists[wbi].append(bit_idx)
        return sorted_msb_lsb_lists
    else:
        print(f"bits_per_weight = {bits_per_weight}")
        assert type(bits_per_weight) == int
        # Convert param ranking to bit ranking
        bit_level_rank = []
        for param in param_ranking:
            bit_idx = param * bits_per_weight
            bit_level_rank.append(bit_idx)

            for j in range(1, bits_per_weight):
                bit_level_rank.append(bit_idx + j)
        bit_lists = [] # List of lists of bit indices
        for i in range(bits_per_weight):
            bit_group = bit_level_rank[i::bits_per_weight]
            bit_lists.append(bit_group)
        return bit_lists 

###################################################################################################


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    #STEP: Load the desired model
    model = load_model(config, pretrained_model=args.pretrained_model)
    print(model.summary())
    print()

    # STEP: Load the dataset
    X_train, y_train, X_test, y_test = load_data()
    assert X_train.shape == (45415, 13)
    assert X_test.shape == (11113, 13)
    assert y_train.shape == (45415, 1)
    assert y_test.shape == (11113, 1)

    #STEP: Process layer precision info, if any
    layer_precision_info = None
    if args.layer_precision_info != None:
        layer_precision_info = eval(args.layer_precision_info[1:-1])

    #STEP: Compute Hessian parameter ranking
    hess = HessianMetrics(
        model, 
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
        X_test, 
        y_test,
        batch_size=1024,
    )

    # Hessian model-wide sensitivity ranking
    eigenvalues, eigenvectors = hess.top_k_eigenvalues(k=8, max_iter=500, rank_BN=False)

    hess_ranking, _ = hess.hessian_ranking_general(
        eigenvectors, eigenvalues=eigenvalues, k=8,
    )

    #STEP: Load the dataset to be used for fic evaluation
    x_test_pred_correct, _ = gen_smart_pix_0mispredicts_dataset(model, X_test, y_test)

    #STEP: If needed, update fault injection bit range args to safe defaults
    nmpb = fk.fmodel.FModelAlt(model, incl_biases=True).num_model_param_bits
    args.fic_range_start = 0    if args.fic_range_start is None else args.fic_range_start
    args.fic_range_stop  = nmpb if args.fic_range_stop  is None else args.fic_range_stop
    args.fic_range_step  = 1    if args.fic_range_step  is None else args.fic_range_step

    model = load_model(config, pretrained_model=args.pretrained_model)

    #STEP: Create fault injection campaign (fic) configuration
    fic_config = {
        "load_model_tuple": (config, args.pretrained_model),
        "model_id" : args.model_id,
        "model"  : model,
        "X_test": x_test_pred_correct,
        "eval_metric_func": my_eval_metric_00,
        "alert_func": my_alert_func_00,
        "fic_output_dir" : args.fic_output_dir,
        "fic_range" : (args.fic_range_start, args.fic_range_stop, args.fic_range_step),
        "bit_width" : args.bit_width,
        "layer_precision_info" : layer_precision_info,
        "hess_ranking" : hess_ranking,
        "last_k": args.last_k,
    }

    #STEP: Launch fic
    fic(fic_config)

###################################################################################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', 
        '--config', 
        type=str, 
        default="./dense_baseline_fkeras_final/fkeras_dense_model_58.h5",
        help="specify yaml config"
    )
    parser.add_argument(
        "--pretrained-model",
        type=str,
        default=None,
        help="specify pretrained model file path",
    )
    parser.add_argument(
        '--model_id', 
        type=str, 
        default="dense_baseline",
        help="specify model id string"
    )
    parser.add_argument(
        '--fic_output_dir', 
        type=str, 
        default=os.getcwd(),
        help="specify output directory for fault injection campaign (fic) files"
    )
    parser.add_argument(
        '--fic_range_start', 
        type=int, 
        default=None,
        help="specify starting bit index (inclusive) for fault injection campaign range(START, stop, step) [Default = 0]"
    )
    parser.add_argument(
        '--fic_range_stop', 
        type=int, 
        default=None,
        help="specify stoping bit index (exclusive) for fault injection campaign range(start, STOP, step) [Default = FModel.num_model_param_bits]"
    )
    parser.add_argument(
        '--fic_range_step', 
        type=int, 
        default=None,
        help="specify step size for fault injection campaign range(start, stop, STEP) [Default = 1]"
    )
    parser.add_argument(
        "--bit_width",
        type=int,
        default=None,
        help="Bitwidth of the weights and biases (assuming single precision)",
    )
    parser.add_argument(
        "--layer_precision_info",
        type=str,
        default=None,
        help="List of tuples describing precision information for each layer (assuming mixed precision)",
    )
    parser.add_argument(
        "--last_k",
        type=int,
        default=5,
        help="Number of last measurements to use for sensitivity pointer calculation",
    )

    args = parser.parse_args()

    main(args)
