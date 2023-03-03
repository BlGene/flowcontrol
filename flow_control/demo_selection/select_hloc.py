import shutil
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from hloc import extract_features, match_features, pairs_from_exhaustive
from flow_control.demo.playback_env_servo import PlaybackEnvServo
from flow_control.localize.hloc_utils import export_images_by_parts
from flow_control.localize.hloc_utils import save_features_seg
from flow_control.localize.hloc_utils import align_pointclouds
from flow_control.localize.hloc_utils import get_playback


class SelectionHloc:
    def __init__(self, root_dir):
        root_dir = Path(root_dir)
        self.parts_fn = root_dir / 'parts.json'
        hloc_root = root_dir.parent / ( str(root_dir.name) + '_hloc')

        self.mapping_dir = hloc_root / 'mapping'
        outputs = hloc_root / 'outputs'
        self.features_path = outputs / 'features.h5'
        self.features_seg_path = outputs / 'features_seg.h5'
        self.matches_path = outputs / 'matches.h5'

        self.root_dir = root_dir
        self.hloc_root = hloc_root
        self.outputs = outputs

        self.feature_conf = extract_features.confs['superpoint_aachen']
        self.matcher_conf = match_features.confs['superglue']

        self.parts_references = export_images_by_parts(self.root_dir, self.parts_fn,
                                                       self.mapping_dir, export=False)

    def clear(self):
        shutil.rmtree(self.mapping_dir, ignore_errors=True)
        shutil.rmtree(self.outputs, ignore_errors=True)

    def preprocess(self):
        self.clear()
        export_images_by_parts(self.root_dir, self.parts_fn, self.mapping_dir)
        #check that we have all references
        references_all = self.parts_references["locate"]
        print(len(references_all))
        #references_all = [ref for ref_part in parts_references.values() for ref in ref_part]
        references_files = [p.relative_to(self.hloc_root).as_posix() for p in (self.mapping_dir).iterdir()]
        assert len(set(references_all)-set(references_files)) == 0

        extract_features.main(self.feature_conf, self.hloc_root, image_list=references_all,
                              feature_path=self.features_path)
        save_features_seg(self.root_dir, self.features_seg_path, self.features_path, references_all)

    def create_query_image(self, query_cam):
        query_dir = self.hloc_root / "query"
        Path(query_dir).mkdir(parents=True, exist_ok=True)
        image_path_query = query_dir / "live.jpg"
        image_arr = query_cam.get_image()[0]
        Image.fromarray(image_arr).save(image_path_query)
        return image_path_query.relative_to(self.hloc_root).as_posix()

    def find_best_demo(self, name_q, query_cam, references):
        results = {}
        for name_d in tqdm(references):
            if name_q == name_d:
                continue

            results[name_d] = align_pointclouds(self.root_dir, self.matches_path,
                                                self.features_path, self.features_seg_path,
                                                name_q, name_d, query_cam=query_cam)

        results = {k: v for k, v in results.items() if v is not None}
        results_sorted = sorted(results.items(), key=lambda t: -t[1]["num_inliers"])
        name_d_best = results_sorted[0][0]
        res_best = results_sorted[0][1]
        return name_d_best, res_best

    def get_best_demo(self, query_cam, part_name='locate'):
        # get image features, references, and loc pairs
        query = self.create_query_image(query_cam)
        extract_features.main(self.feature_conf, self.hloc_root, image_list=[query],
                              feature_path=self.features_path, overwrite=True)
        references_live = self.parts_references[part_name]
        loc_pairs = self.outputs / 'pairs-loc.txt'
        pairs_from_exhaustive.main(loc_pairs, image_list=[query], ref_list=references_live)
        # run matches
        match_features.main(self.matcher_conf, loc_pairs, features=self.features_path,
                            matches=self.matches_path, overwrite=True)

        name_best, res_best = self.find_best_demo(query, query_cam, references_live)
        return name_best, res_best


if __name__ == "__main__":
    root_dir = Path("/home/argusm/CLUSTER/robot_recordings/flow/recombination/2023-01-24")
    selection_hloc = SelectionHloc(root_dir)
    selection_hloc.preprocess()

    name_q = selection_hloc.parts_references['locate'][0]
    pb, frame_index = get_playback(root_dir, name_q)
    query_cam = pb[frame_index].cam
    name_best, res_best = selection_hloc.get_best_demo(query_cam)
    print(name_best)
