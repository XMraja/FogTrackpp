import sys
import os
from multiprocessing import freeze_support
import numpy as np

# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from TrackEval import trackeval  # noqa: E402


def eval(dataset,eval_dir,seqmap,exp_name,fps_div,half_eval = False):
    freeze_support()
    metrics_config = {'METRICS': ['HOTA', 'CLEAR', 'Identity'], 'THRESHOLD': 0.5}

    # TRACKERS_FOLDER = os.getcwd()+"\\eval\\"

    eval_config = {
            'USE_PARALLEL': False,
            'NUM_PARALLEL_CORES': 1,
            'BREAK_ON_ERROR': True,  # Raises exception and exits with error
            'RETURN_ON_ERROR': True,  # if not BREAK_ON_ERROR, then returns from function on error
            'LOG_ON_ERROR': 'None',  # if not None, save any errors into a log file.
            'PRINT_RESULTS': True,
            'PRINT_ONLY_COMBINED': False,
            'PRINT_CONFIG': True,
            'TIME_PROGRESS': False,
            'DISPLAY_LESS_PROGRESS': True,
            'OUTPUT_SUMMARY': True,
            'OUTPUT_EMPTY_CLASSES': True,  # If False, summary files are not output for classes with no detections
            'OUTPUT_DETAILED': True,
            'PLOT_CURVES': True,
        }
    
    if half_eval == False:
        gt_format = '{gt_folder}/{seq}/gt/gt.txt'
    elif fps_div == 1:
        gt_format = '{gt_folder}/{seq}/gt/gt_val_half.txt'
    else:
        gt_format = '{gt_folder}/{seq}/gt/gt_1_'+f'{fps_div}.txt'

    dataset_config = {
            'GT_FOLDER': dataset,  # Location of GT data
            'TRACKERS_FOLDER': eval_dir,  # Trackers location
            'OUTPUT_FOLDER': None,  # Where to save eval results (if None, same as TRACKERS_FOLDER)
            'TRACKERS_TO_EVAL': [exp_name],  # Filenames of trackers to eval (if None, all in folder)
            

            'CLASSES_TO_EVAL': ['vessel'],  # Valid: ['pedestrian']
            'BENCHMARK': 'MOT20',  # Valid: 'MOT17', 'MOT16', 'MOT20', 'MOT15'
            'SPLIT_TO_EVAL': 'val',  # Valid: 'train', 'test', 'all'
            'INPUT_AS_ZIP': False,  # Whether tracker input files are zipped
            'PRINT_CONFIG': False,  # Whether to print current config
            'DO_PREPROC': True,  # Whether to perform preprocessing (never done for MOT15)
            'TRACKER_SUB_FOLDER': '',  # Tracker files are in TRACKER_FOLDER/tracker_name/TRACKER_SUB_FOLDER
            'OUTPUT_SUB_FOLDER': '',  # Output files are saved in OUTPUT_FOLDER/tracker_name/OUTPUT_SUB_FOLDER
            'TRACKER_DISPLAY_NAMES': None,  # Names of trackers to display, if None: TRACKERS_TO_EVAL
            'SEQMAP_FOLDER': None,  # Where seqmaps are found (if None, GT_FOLDER/seqmaps)
            'SEQMAP_FILE': seqmap,  # Directly specify seqmap file (if none use seqmap_folder/benchmark-split_to_eval)
            'SEQ_INFO': None,  # If not None, directly specify sequences to eval and their number of timesteps
            'GT_LOC_FORMAT': gt_format,  
            'SKIP_SPLIT_FOL': True
        }


    # Run code
    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity, trackeval.metrics.VACE]:
        if metric.get_name() in metrics_config['METRICS']:
            metrics_list.append(metric(metrics_config))
    if len(metrics_list) == 0:
        raise Exception('No metrics selected for evaluation')
    output_res, output_msg = evaluator.evaluate(dataset_list, metrics_list)
    
    return output_res['summary'][0]['HOTA'], output_res['summary'][2]['IDF1'], output_res['summary'][1]['MOTA'], output_res['summary'][0]['AssA']


def main():
    # 设置路径
    dataset_path = "./datasets/vesselmot/test"
    eval_dir_path = "./Yolov7-tracker-2.1/track_results/sort/mot17/test"
    seqmap_path = "./datasets/vesselmot/seqmaps/vesselmot-test-all.txt"
    
    # 运行评估
    try:
        HOTA, IDF1, MOTA, AssA = eval(
            dataset=dataset_path,
            eval_dir=eval_dir_path,
            seqmap=seqmap_path,
            exp_name="sort",
            fps_div=1,
            half_eval=False
        )
        
        print("评估结果:")
        print(f"HOTA: {HOTA:.3f}")
        print(f"IDF1: {IDF1:.3f}") 
        print(f"MOTA: {MOTA:.3f}")
        print(f"AssA: {AssA:.3f}")
        
    except Exception as e:
        print(f"评估过程中出现错误: {e}")

if __name__ == "__main__":
    main()

    # /home/user/.conda/envs/mot_yolox/bin/python eval_for_yolox.py