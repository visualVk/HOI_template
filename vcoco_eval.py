from utils.vsrl_eval import VCOCOeval
import utils.vcoco_cached_helper as vhelper

if __name__ == "__main__":
    vsrl_annot_file = "data/vcoco/vcoco_test.json"
    coco_file = "data/instances_vcoco_all_2014.json"
    split_file = "data/splits/vcoco_test.ids"

    # Change this line to match the path of your cached file
    det_file = "./data/cache.pkl"

    print(f"Loading cached results from {det_file}.")
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)