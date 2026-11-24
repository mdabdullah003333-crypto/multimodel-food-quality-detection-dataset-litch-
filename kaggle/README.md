# Kaggle run guide

1. Upload LADD as a Kaggle Dataset (https://www.kaggle.com/datasets -> New Dataset).
   Zip must contain: Litchi_Sensor_Data.csv and images/{Fresh,Rotten,Formalin}/...
2. New Notebook -> Settings -> Accelerator -> GPU T4 x2 (or P100).
3. (Optional manual upload) Upload LADD as a Kaggle Dataset instead, then note its
   slug shown under /kaggle/input/<your-dataset-name> - the script auto-detects it.
4. Paste train_ladd_tf.py into a cell and run. DATA auto-resolves: it probes
   /kaggle/input/ladd-dataset, /kaggle/working/ladd, ~/ladd, then the working dir,
   and takes the first location containing Litchi_Sensor_Data.csv.
   To force a custom path, set DATA = "/kaggle/input/<your-dataset-name>" before
   the resolve_data() call. Outputs (model, tflite, log, fv_cache) land in
   /kaggle/working/.
5. For figures: run make_thesis_figures.py in the same session (needs tf_train_log.txt,
   hybrid_freshness.keras, hybrid_freshness_int8.tflite already in /kaggle/working/);
   figures appear in /kaggle/working/thesis_figures/.
