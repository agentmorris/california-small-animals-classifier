#%% Header

"""

Notebook for exploring the results of a validation pass.

"""

#%% Imports and constants

import os
import json

image_folder = 'f:/data/california-small-animals-training/val'
results_file = 'c:/temp/california-small-animals-output/runs/eva02-20260608/val_predictions_eva02-20260608-00.json'
assert os.path.isdir(image_folder)
assert os.path.isfile(results_file)


#%% Validate results file

from megadetector.postprocessing.validate_batch_results import \
    ValidateBatchResultsOptions, validate_batch_results

options = ValidateBatchResultsOptions()

options.check_image_existence = True
options.relative_path_base = image_folder
options.return_data = False
options.verbose = False
options.raise_errors = True

r = validate_batch_results(json_filename=results_file,
                           options=options)


#%% Prepare ground truth file

from megadetector.utils.path_utils import find_images
from megadetector.utils.ct_utils import write_json

val_json_file = os.path.join(image_folder,'val_cct.json')

with open(results_file,'r') as f:
    results = json.load(f)

classification_category_names = set(results['classification_categories'].values())

category_folders = os.listdir(image_folder)
category_folders = \
    [s for s in category_folders if os.path.isdir(os.path.join(image_folder,s))]

assert set(category_folders) == classification_category_names

image_files_relative = find_images(image_folder,return_relative_paths=True,recursive=True)

images = []
annotations = []

category_name_to_id = {}
categories = []

for i_category,category_name in enumerate(classification_category_names):
    category_id = i_category
    category_name_to_id[category_name] = category_id
    category = {'name':category_name,'id':category_id}
    categories.append(category)

for fn in image_files_relative:

    tokens = fn.split('/')
    category_name = tokens[0]
    assert category_name in classification_category_names

    im = {}
    im['file_name'] = fn
    im['id'] = fn
    images.append(im)

    ann = {}
    ann['image_id'] = fn
    ann['id'] = fn + '_ann'
    ann['category_id'] = category_name_to_id[category_name]
    annotations.append(ann)

coco_out = {}
coco_out['info'] = {}
coco_out['images'] = images
coco_out['annotations'] = annotations
coco_out['categories'] = categories

write_json(val_json_file,coco_out)


#%% Validate COCO file

from megadetector.data_management.databases.integrity_check_json_db import \
    IntegrityCheckOptions, integrity_check_json_db

options = IntegrityCheckOptions()

options.baseDir = image_folder
options.bCheckImageSizes = False
options.bCheckImageExistence = True
options.bFindUnusedImages = True
options.bRequireLocation = False
options.iMaxNumImages = -1
options.nThreads = 10
options.parallelizeWithThreads = True
options.verbose = True
options.allowIntIDs = False
options.requireInfo = True
options.validateBoxes = None

sorted_categories, data, error_info = \
    integrity_check_json_db(json_file=val_json_file,
                            options=options)


#%% Preview COCO file

from megadetector.visualization.visualize_db import visualize_db, DbVizOptions
from megadetector.utils.path_utils import open_file

val_preview_folder = 'c:/temp/csa-val-preview'

options = DbVizOptions()
options.num_to_visualize = 500
options.viz_size = (1000, -1)
options.sort_by_filename = True
options.trim_to_images_with_bboxes = False
options.random_seed = 0
options.add_search_links = False
options.include_image_links = False
options.include_filename_links = False
options.classes_to_include = None
options.classes_to_exclude = None
options.multiple_categories_tag = '*multiple*'
options.parallelize_rendering = True
options.parallelize_rendering_with_threads = True
options.parallelize_rendering_n_cores = 12
options.show_full_paths = False
options.extra_image_fields_to_print = None
options.extra_annotation_fields_to_print = None
options.force_rendering = True
options.verbose = False
options.custom_category_mapping = None
options.quality = None
options.colormap = None
options.create_category_pages = False
options.max_sequence_length = None

html_output_file,image_db = visualize_db(db_path=val_json_file,
                                         output_dir=val_preview_folder,
                                         image_base_dir=image_folder,
                                         options=options)

open_file(html_output_file)


#%% Classification analysis

analysis_preview_folder = os.path.join(os.path.dirname(results_file),'classification-analysis')

from megadetector.postprocessing.analyze_classification_results import \
    ClassificationAnalysisOptions, analyze_classification_results

options = ClassificationAnalysisOptions()

options.results_file = results_file
options.gt_file = val_json_file
options.classification_confidence_threshold = 0.0

options.detection_threshold = 0.0
options.image_base_dir = image_folder
options.html_output_dir = analysis_preview_folder
options.max_total_images = 8000
options.max_images_per_cell = 50
options.random_seed = 0
options.detection_category_mapping = None
options.apply_detection_category_mapping_when_classifications_are_present = True
options.sequence_level_analysis = False
options.rendering_workers = 10
options.rendering_pool_type = 'threads'
options.overwrite = True
options.show_overall_metrics = True
options.output_image_width = 1000
options.n_mispredictions_for_table = 5
options.categories_to_ignore = None
options.single_prediction_per_image = False
options.single_label_per_image = False
options.max_images_per_html_file = 1000
options.predicted_category_name_mappings = None
options.gt_category_name_mappings = None

r = analyze_classification_results(options)
open_file(r.html_output_file)
