import os
import yaml
import time
import pickle
import argparse
import numpy as np
import fkeras as fk
import pandas as pd
from tqdm import tqdm
from datetime import datetime

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

def gen_smart_pix_0mispredicts_dataset(model):
    X_train, y_train, X_test, y_test = load_data()
    assert X_train.shape == (45415, 13)
    assert X_test.shape == (11113, 13)
    assert y_train.shape == (45415, 1)
    assert y_test.shape == (11113, 1)

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
    
    print(f"fic_range = {fic_config['fic_range']}")
    range_list = range(*fic_config['fic_range'])
    print(f"Range list: {range_list}")

    fi_times = list()
    for bit_i in tqdm(range(*fic_config["fic_range"]) ):
        #STEP: Indicate the current bit being flipped in info log
        # print(f"[fic] Injecting bit flip at global-bit-index: {bit_i}\n", file=fic_log_info)

        curr_fi_start_time = time.time()
        #STEP: Flip the desired bit(s) in the "original" model to make it "faulty"
        fmodel.explicitly_flip_bits([bit_i])

        #STEP: Compute and save the provided metric for the faulty model
        y_pred_fault = fmodel.model.predict(fic_config["X_test"], verbose=0)
        curr_fi_time = time.time() - curr_fi_start_time
        fi_times.append(curr_fi_time)
        metric_value = fic_config["eval_metric_func"](y_pred, y_pred_fault)
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
            alerts.append(bit_i)
            fic_log_info.flush()

        fmodel.explicitly_reset_bits([bit_i])

    avg_fi_time = np.mean(fi_times)
    print(f"[fic] Average time per bit flip: {avg_fi_time}", file=fic_log_info)
    print(f"[fic] Average time per bit flip: {avg_fi_time}")

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


def main(args):
    with open(args.config) as f:
        config = yaml.safe_load(f)
    #STEP: Load the desired model
    model = load_model(config, pretrained_model=args.pretrained_model)
    print(model.summary())
    print()

    #STEP: Load the dataset to be used for evaluation
    x_test_pred_correct, _ = gen_smart_pix_0mispredicts_dataset(model)


    #STEP: If needed, update fault injection bit range args to safe defaults
    nmpb = fk.fmodel.FModelAlt(model, incl_biases=True).num_model_param_bits
    args.fic_range_start = 0    if args.fic_range_start is None else args.fic_range_start
    args.fic_range_stop  = nmpb if args.fic_range_stop  is None else args.fic_range_stop
    args.fic_range_step  = 1    if args.fic_range_step  is None else args.fic_range_step


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

    args = parser.parse_args()

    main(args)
